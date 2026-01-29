"""
VQ + XYZ quantization combined codec interface with merged mask.

This module provides the combined VQ + XYZ codec interface that uses a single
merged mask for interframe encoding.
"""

from ..interframe.combine import (
    CombinedInterframeCodecInterface,
    CombinedInterframeCodecContext,
    CombinedPayload,
)
from ..vq.interface import VQInterframeCodecContext
from ..vq.singlemask import VQMergeMaskInterframeCodecInterface, VQMergeMaskInterframePayload
from ..xyz.interface import (
    XYZQuantInterframeCodecInterface,
    XYZQuantInterframeCodecContext,
    XYZQuantInterframePayload,
)


class VQXYZQuantMergeMaskInterframeCodecInterface(CombinedInterframeCodecInterface):
    """
    Combined VQ + XYZ codec with single merged mask for interframe encoding.

    This interface extends CombinedInterframeCodecInterface but uses a single merged mask
    (OR of XYZ mask and all VQ attribute masks) for interframe payloads. Both XYZ and VQ
    payloads use the same mask, improving compression efficiency.

    Assumes interfaces order: [xyz_interface, vq_interface]
    """

    def __init__(self):
        """
        Initialize the combined codec with XYZ and VQ interfaces.
        """
        xyz_interface = XYZQuantInterframeCodecInterface()
        vq_interface = VQMergeMaskInterframeCodecInterface()
        super().__init__([xyz_interface, vq_interface])
        self.xyz_interface = xyz_interface
        self.vq_interface = vq_interface

    def encode_interframe(
        self,
        prev_context: CombinedInterframeCodecContext,
        next_context: CombinedInterframeCodecContext,
    ) -> CombinedPayload:
        """
        Encode the difference between two consecutive frames.

        Calls super to get individual payloads, then merges masks (OR) and
        re-extracts data using the merged mask.

        Args:
            prev_context: The context of the previous frame.
            next_context: The context of the next frame.

        Returns:
            A CombinedPayload with XYZ and VQ payloads using the same merged mask.
        """
        # Get original payloads from parent
        original_payload: CombinedPayload = super().encode_interframe(prev_context, next_context)
        xyz_original: XYZQuantInterframePayload = original_payload.payloads[0]
        vq_original: VQMergeMaskInterframePayload = original_payload.payloads[1]

        # Merge masks: XYZ mask OR VQ mask
        merged_mask = xyz_original.xyz_mask | vq_original.ids_mask

        # Get next context data
        xyz_next_context: XYZQuantInterframeCodecContext = next_context.contexts[0]
        vq_next_context: VQInterframeCodecContext = next_context.contexts[1]

        # Create XYZ payload with merged mask
        xyz_payload = XYZQuantInterframePayload(
            xyz_mask=merged_mask,
            quantized_xyz=xyz_next_context.quantized_xyz[merged_mask],
        )

        # Create VQ payload with merged mask
        changed_ids_dict = {}
        for key, next_ids in vq_next_context.ids_dict.items():
            changed_ids_dict[key] = next_ids[merged_mask]

        vq_payload = VQMergeMaskInterframePayload(
            ids_mask=merged_mask,
            ids_dict=changed_ids_dict,
        )

        return CombinedPayload(payloads=[xyz_payload, vq_payload])
