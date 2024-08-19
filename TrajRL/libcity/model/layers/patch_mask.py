import torch
import torch.nn as nn


class Patcher:

    def __init__(self, patch_len, stride):
        """

        Args:
            patch_len:        patcher length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def patch(self, xb_emb):
        """
        take xb from learner and convert to patcher: [bs x seq_len x 1] -> [bs x num_patch x 1 x patch_len]
        """
        xb_patch, num_patch = create_patch(xb_emb, self.patch_len, self.stride)

        return xb_patch


class PatchMasker:
    def __init__(self, patch_len, stride, mask_ratio):
        """
        Args:
            patch_len:        patcher length
            stride:           stride
            mask_ratio:       mask ratio
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio

    def patch_masking(self, x):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """

        # xb_label = xb[..., 0]  # [bs x seq_len x n_vars] n_vars == 1
        # xb_label = xb  # [bs x seq_len x n_vars] n_vars == 1
        xb_patch, num_patch = create_patch2(x, self.patch_len, self.stride)
        # xb_patch: [bs x num_patch x n_vars x d_model x patch_len] xb_label: [bs x num_patch x n_vars x patch_len]

        xb_patch_mask, _, mask, _ = random_masking(xb_patch, self.mask_ratio)
        # xb_mask: [bs x num_patch x n_vars x d_model x patch_len]
        mask = mask.bool()  # mask: [bs x num_patch x n_vars]

        return xb_patch_mask, mask


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars x d_model]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len

    xb = xb[:, s_begin:, :, :]  # xb: [bs x tgt_len x 1]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)  # xb: [bs x num_patch x 1 x d_model x patch_len]

    xb = xb.permute(0, 1, 2, 4, 3)  # xb: [bs x num_patch x 1 x patch_len x d_model]

    # xb_label = xb_label[:, s_begin:, :]
    # xb_label = xb_label.unfold(dimension=1, size=tgt_len,
    # step=stride)  # xb_label: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


def create_patch2(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len

    xb = xb[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)  # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1,
                                                                        D))  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, nvars, D,
                            device=xb.device)  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1,
                                                                              D))  # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


# def random_masking(xb, mask_ratio):
#     # xb: [bs x num_patch x n_vars x patch_len x D]
#     bs, L, nvars, patch_len, D = xb.shape
#     x = xb.clone()
#
#     len_keep = int(L * (1 - mask_ratio))
#
#     noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars
#
#     # sort noise for each sample
#     ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is removed
#     ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]
#
#     # keep the first subset
#     ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
#
#     x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, patch_len,
#                                                                                       D))  # x_kept: [bs x len_keep x nvars  x patch_len x embedding_size]
#
#     # removed x
#     x_removed = torch.zeros(bs, L - len_keep, nvars, patch_len, D,
#                             device=xb.device)  # x_removed: [bs x (L-len_keep) x nvars x patch_len x embedding_size]
#     x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len x embedding_size]
#
#     # combine the kept part and the removed one
#     x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, patch_len,
#                                                                                       D))  # x_masked: [bs x num_patch x nvars x patch_len x embedding_size]
#
#     # generate the binary mask: 0 is keep, 1 is remove
#     mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
#     mask[:, :len_keep, :] = 0
#     # unshuffle to get the binary mask
#     mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
#     return x_masked, x_kept, mask, ids_restore
