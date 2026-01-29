from dataclasses import dataclass
from typing import Dict, Self

import torch

from ..payload import Payload
from .interface import (
    VQInterframeCodecInterface,
    VQInterframeCodecContext,
)


@dataclass
class VQMergeMaskInterframePayload(Payload):
    """
    Payload for VQ interframe data with a single merged mask.

    Contains a single mask that is the OR of all individual attribute masks,
    and all ids arrays have the same length (based on the merged mask).

    Attributes:
        ids_mask: Boolean tensor indicating which positions changed (OR of all masks).
        ids_dict: Dict of changed ids values, all with the same length.
    """
    ids_mask: torch.Tensor
    ids_dict: Dict[str, torch.Tensor]

    def to(self, device) -> Self:
        """
        Move the Payload to the specified device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda', torch.device).

        Returns:
            A new VQMergeMaskInterframePayload instance on the target device.
        """
        return VQMergeMaskInterframePayload(
            ids_mask=self.ids_mask.to(device),
            ids_dict={k: v.to(device) for k, v in self.ids_dict.items()},
        )


class VQMergeMaskInterframeCodecInterface(VQInterframeCodecInterface):
    """
    VQ-based inter-frame encoding/decoding interface with merged mask.

    This interface extends VQInterframeCodecInterface but uses a single merged mask
    (OR of all attribute masks) for the interframe payload. All ids arrays have
    the same length, making serialization and compression more efficient.
    """

    def decode_interframe(
        self,
        payload: VQMergeMaskInterframePayload,
        prev_context: VQInterframeCodecContext,
    ) -> VQInterframeCodecContext:
        """
        Decode a delta payload to reconstruct the next frame's context.

        Applies the changed values from the payload to the previous context
        using the single merged mask.

        Args:
            payload: The delta payload containing changed cluster IDs with merged mask.
            prev_context: The context of the previous frame (contains codebook).

        Returns:
            The reconstructed context for the current frame.
        """
        # Clone previous ids_dict and apply changes using merged mask
        new_ids_dict = {}
        mask = payload.ids_mask
        for key, prev_ids in prev_context.ids_dict.items():
            new_ids = prev_ids.clone()
            new_ids[mask] = payload.ids_dict[key]
            new_ids_dict[key] = new_ids

        return VQInterframeCodecContext(
            ids_dict=new_ids_dict,
            codebook_dict=prev_context.codebook_dict,
            max_sh_degree=prev_context.max_sh_degree,
        )

    def encode_interframe(
        self,
        prev_context: VQInterframeCodecContext,
        next_context: VQInterframeCodecContext,
    ) -> VQMergeMaskInterframePayload:
        """
        Encode the difference between two consecutive frames.

        Computes a merged mask (OR of all individual masks) and extracts
        changed values for all attributes using this single mask.

        Args:
            prev_context: The context of the previous frame.
            next_context: The context of the next frame.

        Returns:
            A payload containing changed cluster IDs with a single merged mask.
        """
        # Compute individual masks and merge them
        merged_mask = None
        for key in next_context.ids_dict.keys():
            prev_ids = prev_context.ids_dict[key]
            next_ids = next_context.ids_dict[key]
            # Compare ids, handling different tensor shapes
            if prev_ids.dim() == 1:
                mask = prev_ids != next_ids
            else:
                mask = (prev_ids != next_ids).any(dim=-1)

            if merged_mask is None:
                merged_mask = mask
            else:
                merged_mask = merged_mask | mask

        # Extract changed ids using merged mask for all keys
        changed_ids_dict = {}
        for key, next_ids in next_context.ids_dict.items():
            changed_ids_dict[key] = next_ids[merged_mask]

        return VQMergeMaskInterframePayload(
            ids_mask=merged_mask,
            ids_dict=changed_ids_dict,
        )
