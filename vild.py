import numpy as np
import torch
import clip
import numpy as np
from PIL import Image
from scipy.special import softmax
import tensorflow.compat.v1 as tf
import cv2
import collections
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


class VILD_CLIP:
    def __init__(self, vild_folder, nms_threshold=0.5, min_rpn_score_thresh=0.9,
            min_box_area=100, min_score=0.6, use_clip_embeddings=True,
            mask_out_cropped_images=True, crop_padding_size=20):
        self.vild_folder = vild_folder
        self.nms_threshold = nms_threshold
        self.min_rpn_score_thresh = min_rpn_score_thresh
        self.min_box_area = min_box_area
        self.min_score = min_score
        self.use_clip_embeddings = use_clip_embeddings
        self.mask_out_cropped_images = mask_out_cropped_images
        self.crop_padding_size = crop_padding_size

        self.model, self.preprocess = clip.load("ViT-B/32")
        self.sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(self.sess, ['serve'], self.vild_folder)

    @staticmethod
    def modify_VLID_graph(vild_folder):
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ['serve'], vild_folder)
            modified_graph_def = tf.GraphDef()
            for node in sess.graph.as_graph_def().node:
                new_node = tf.NodeDef()
                new_node.CopyFrom(node)
                if new_node.name == "Placeholder":
                    new_node.attr['dtype'].type = tf.uint8.as_datatype_enum
                    new_node.attr['shape'].shape.CopyFrom(tf.TensorShape((None, None, 3)).as_proto())
                elif new_node.name == "Squeeze":
                    continue
                elif new_node.name == "ReadFile":
                    continue
                elif new_node.name == "decode_image/DecodeImage":
                    continue
                elif new_node.name == "convert_image/Cast":
                    new_node.input[0] = "Placeholder"
                modified_graph_def.node.extend([new_node])
        return modified_graph_def

    def run(self, image_file, categories, reject_categories=tuple()):
        categories = np.array(categories)
        roi_boxes, roi_scores, detection_boxes, scores_unused, \
        box_outputs, detection_masks, visual_features, image_info = self.sess.run([
            'RoiBoxes:0', 'RoiScores:0', '2ndStageBoxes:0', '2ndStageScoresUnused:0',
            'BoxOutputs:0', 'MaskOutputs:0', 'VisualFeatOutputs:0', 'ImageInfo:0'],
                feed_dict={'Placeholder:0': [image_file,]})

        roi_boxes = np.squeeze(roi_boxes, axis=0)
        roi_scores = np.squeeze(roi_scores, axis=0)
        detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
        scores_unused = np.squeeze(scores_unused, axis=0)
        box_outputs = np.squeeze(box_outputs, axis=0)
        detection_masks = np.squeeze(detection_masks, axis=0)
        visual_features = np.squeeze(visual_features, axis=0)
        image_info = np.squeeze(image_info, axis=0)

        image_scale = np.tile(image_info[2:3, :], (1, 2))
        image_height = int(image_info[0, 0])
        image_width = int(image_info[0, 1])

        rescaled_detection_boxes = detection_boxes / image_scale
        boxes_areas = \
            (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * \
            (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

        nms_indices = self._nms(detection_boxes, roi_scores)
        num = len(roi_boxes)
        indices = np.where(
            (np.isin(np.arange(num, dtype=int), nms_indices)) &
            (np.any(roi_boxes != 0, axis=-1)) &
            (roi_scores >= self.min_rpn_score_thresh) &
            (boxes_areas >= self.min_box_area))[0]

        roi_scores = roi_scores[indices]
        detection_boxes = detection_boxes[indices]
        detection_masks = detection_masks[indices]
        visual_features = visual_features[indices]
        rescaled_detection_boxes = rescaled_detection_boxes[indices]

        ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
        processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
        segmentations = VILD_CLIP._paste_instance_masks(
            detection_masks, processed_boxes, image_height, image_width)

        if self.use_clip_embeddings:
            image = np.asarray(Image.open(open(image_file, 'rb')).convert("RGB"))
            visual_features = self._build_visual_embeddings(
                image, rescaled_detection_boxes, segmentations)

        text_features = self._build_text_embeddings(categories)
        scores = np.matmul(visual_features, text_features.T)
        scores *= 100
        scores = softmax(scores, axis=-1)

        selected_indices = np.where(
            (np.max(scores, axis=1) >= self.min_score) &
            (np.logical_not(np.isin(categories[np.argmax(scores, axis=1)], reject_categories))))[0]

        roi_scores = roi_scores[selected_indices]
        detection_boxes = detection_boxes[selected_indices]
        detection_masks = detection_masks[selected_indices]
        visual_features = visual_features[selected_indices]
        rescaled_detection_boxes = rescaled_detection_boxes[selected_indices]
        segmentations = segmentations[selected_indices]
        scores = scores[selected_indices]

        return rescaled_detection_boxes, roi_scores, segmentations, scores

    def _build_text_embeddings(self, text):
        with torch.no_grad():
            tokens = clip.tokenize(text)
            tokens = tokens.cuda()
            embeddings = self.model.encode_text(tokens)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.cpu().numpy()
        return embeddings

    def _build_visual_embeddings(self, image,
            rescaled_detection_boxes, segmentations):
        image_height = image.shape[0]
        image_width = image.shape[1]
        background_image_height = image_height + self.crop_padding_size * 2
        background_image_width = image_width + self.crop_padding_size * 2
        background_image = np.zeros((background_image_height, background_image_width, 3),
            dtype=np.uint8)
        background_image[
            self.crop_padding_size:image_height+self.crop_padding_size,
            self.crop_padding_size:image_width+self.crop_padding_size] = image
        for mask in segmentations:
            background_image[
                self.crop_padding_size:image_height+self.crop_padding_size,
                self.crop_padding_size:image_width+self.crop_padding_size][mask != 0] = 0

        with torch.no_grad():
            prepared_crops = list()
            for bbox, mask in zip(rescaled_detection_boxes, segmentations):
                y1 = max(0, int(np.floor(bbox[0])))
                x1 = max(0, int(np.floor(bbox[1])))
                y2 = min(image_height, int(np.ceil(bbox[2])))
                x2 = min(image_width, int(np.ceil(bbox[3])))
                crop = np.copy(image[y1:y2, x1:x2])
                crop_with_background = background_image[
                    y1:y2+self.crop_padding_size*2,
                    x1:x2+self.crop_padding_size*2].copy()
                crop_with_background[
                    self.crop_padding_size:(y2-y1)+self.crop_padding_size,
                    self.crop_padding_size:(x2-x1)+self.crop_padding_size][mask[y1:y2, x1:x2] != 0] = \
                        crop[mask[y1:y2, x1:x2] != 0]
                prepared_crop = self.preprocess(Image.fromarray(crop_with_background)).cuda()
                prepared_crops.append(prepared_crop)
            prepared_crops = torch.stack(prepared_crops, dim=0)
            embeddings = self.model.encode_image(prepared_crops).cpu().numpy()
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def _nms(self, dets, scores, max_dets=1000):
        """ Non-maximum suppression.
        Args:
            dets: [N, 4]
            scores: [N,]
            thresh: iou threshold. Float
            max_dets: int.
        """
        y1 = dets[:, 0]
        x1 = dets[:, 1]
        y2 = dets[:, 2]
        x2 = dets[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0 and len(keep) < max_dets:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            overlap = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-12)

            inds = np.where(overlap <= self.nms_threshold)[0]
            order = order[inds + 1]
        return keep

    @staticmethod
    def draw_detections(image, rescaled_detection_boxes, segmentations,
            roi_scores=None, scores=None):
        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        box_to_instance_masks_map = {}
        box_to_score_map = {}

        for i in range(rescaled_detection_boxes.shape[0]):
            box = tuple(rescaled_detection_boxes[i].tolist())
            box_to_instance_masks_map[box] = segmentations[i]
            box_to_color_map[box] = 'red'
            box_to_score_map[box] = roi_scores[i]
            box_to_display_str_map[box] = [
                f"roi score: {roi_scores[i] * 100:.02f}%, "
                f"category score: {np.max(scores[i]) * 100:.02f}%"]

        box_color_iter = sorted(
            box_to_color_map.items(), key=lambda kv: box_to_score_map[kv[0]])

        for box, color in box_color_iter:
            ymin, xmin, ymax, xmax = box
            VILD_CLIP._draw_mask_on_image_array(image, box_to_instance_masks_map[box],
                color=color, alpha=0.4)
            VILD_CLIP._draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax,
                color=color, thickness=4, display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=False)

    @staticmethod
    def _expand_boxes(boxes, scale):
        """ Expands an array of boxes by a given scale. """
        # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L227  # pylint: disable=line-too-long
        # The `boxes` in the reference implementation is in [x1, y1, x2, y2] form,
        # whereas `boxes` here is in [x1, y1, w, h] form
        w_half = boxes[:, 2] * .5
        h_half = boxes[:, 3] * .5
        x_c = boxes[:, 0] + w_half
        y_c = boxes[:, 1] + h_half

        w_half *= scale
        h_half *= scale

        boxes_exp = np.zeros(boxes.shape)
        boxes_exp[:, 0] = x_c - w_half
        boxes_exp[:, 2] = x_c + w_half
        boxes_exp[:, 1] = y_c - h_half
        boxes_exp[:, 3] = y_c + h_half

        return boxes_exp

    @staticmethod
    def _paste_instance_masks(masks, detected_boxes, image_height, image_width):
        """ Paste instance masks to generate the image segmentation results.

        Args:
            masks: a numpy array of shape [N, mask_height, mask_width] representing the
            instance masks w.r.t. the `detected_boxes`.
            detected_boxes: a numpy array of shape [N, 4] representing the reference
            bounding boxes.
            image_height: an integer representing the height of the image.
            image_width: an integer representing the width of the image.

        Returns:
            segmentations: a numpy array of shape [N, image_height, image_width] representing
            the instance masks *pasted* on the image canvas.
        """

        # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test.py#L812  # pylint: disable=line-too-long
        # To work around an issue with cv2.resize (it seems to automatically pad
        # with repeated border values), we manually zero-pad the masks by 1 pixel
        # prior to resizing back to the original image resolution. This prevents
        # "top hat" artifacts. We therefore need to expand the reference boxes by an
        # appropriate factor.
        _, mask_height, mask_width = masks.shape
        scale = max((mask_width + 2.0) / mask_width,
                    (mask_height + 2.0) / mask_height)

        ref_boxes = VILD_CLIP._expand_boxes(detected_boxes, scale)
        ref_boxes = ref_boxes.astype(np.int32)
        padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
        segmentations = []
        for mask_ind, mask in enumerate(masks):
            im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            # Process mask inside bounding boxes.
            padded_mask[1:-1, 1:-1] = mask[:, :]

            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > 0.5, dtype=np.uint8)

            x_0 = min(max(ref_box[0], 0), image_width)
            x_1 = min(max(ref_box[2] + 1, 0), image_width)
            y_0 = min(max(ref_box[1], 0), image_height)
            y_1 = min(max(ref_box[3] + 1, 0), image_height)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                (x_0 - ref_box[0]):(x_1 - ref_box[0])
            ]
            segmentations.append(im_mask)

        segmentations = np.array(segmentations)
        assert masks.shape[0] == segmentations.shape[0]
        return segmentations

    @staticmethod
    def _draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
        """Draws mask on an image.

        Args:
            image: uint8 numpy array with shape (img_height, img_height, 3)
            mask: a uint8 numpy array of shape (img_height, img_height) with
            values between either 0 or 1.
            color: color to draw the keypoints with. Default is red.
            alpha: transparency value between 0 and 1. (default: 0.4)

        Raises:
            ValueError: On incorrect data type for image or masks.
        """
        if image.dtype != np.uint8:
            raise ValueError('`image` not of type np.uint8')
        if mask.dtype != np.uint8:
            raise ValueError('`mask` not of type np.uint8')
        if np.any(np.logical_and(mask != 1, mask != 0)):
            raise ValueError('`mask` elements should be in [0, 1]')
        if image.shape[:2] != mask.shape:
            raise ValueError('The image has spatial dimensions %s but the mask has '
                            'dimensions %s' % (image.shape[:2], mask.shape))
        rgb = ImageColor.getrgb(color)
        pil_image = Image.fromarray(image)

        solid_color = np.expand_dims(
            np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        np.copyto(image, np.array(pil_image.convert('RGB')))

    @staticmethod
    def _draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax,
            color='red', thickness=4, display_str_list=(), use_normalized_coordinates=True):
        """Adds a bounding box to an image (numpy array).

        Bounding box coordinates can be specified in either absolute (pixel) or
        normalized coordinates by setting the use_normalized_coordinates argument.

        Args:
            image: a numpy array with shape [height, width, 3].
            ymin: ymin of bounding box.
            xmin: xmin of bounding box.
            ymax: ymax of bounding box.
            xmax: xmax of bounding box.
            color: color to draw bounding box. Default is red.
            thickness: line thickness. Default value is 4.
            display_str_list: list of strings to display in box
                            (each to be shown on its own line).
            use_normalized_coordinates: If True (default), treat coordinates
            ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
            coordinates as absolute.
        """
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        VILD_CLIP._draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                                    thickness, display_str_list,
                                    use_normalized_coordinates)
        np.copyto(image, np.array(image_pil))

    @staticmethod
    def _draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,
            color='red', thickness=4, display_str_list=(), use_normalized_coordinates=True):
        """Adds a bounding box to an image.

        Bounding box coordinates can be specified in either absolute (pixel) or
        normalized coordinates by setting the use_normalized_coordinates argument.

        Each string in display_str_list is displayed on a separate line above the
        bounding box in black text on a rectangle filled with the input 'color'.
        If the top of the bounding box extends to the edge of the image, the strings
        are displayed below the bounding box.

        Args:
            image: a PIL.Image object.
            ymin: ymin of bounding box.
            xmin: xmin of bounding box.
            ymax: ymax of bounding box.
            xmax: xmax of bounding box.
            color: color to draw bounding box. Default is red.
            thickness: line thickness. Default value is 4.
            display_str_list: list of strings to display in box
                            (each to be shown on its own line).
            use_normalized_coordinates: If True (default), treat coordinates
            ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
            coordinates as absolute.
        """
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        if use_normalized_coordinates:
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], width=thickness, fill=color)
        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except IOError:
            font = ImageFont.load_default()

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height
        # Reverse list and print from bottom to top.
        for display_str in display_str_list[::-1]:
            text_left = min(5, left)
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle(
                [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                text_bottom)],
                fill=color)
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
            text_bottom -= text_height - 2 * margin
