import itertools
import logging
import os
import torch
from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    RawLabelDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from cll_merged.data import LanguagePairDatasetCllMerged
EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


def load_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )
    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    # start to deal with direction parameters
    src_direction, tgt_direction = None, None
    # the path judgement is start from main content
    src_direction_path = os.path.join("directionFiles", "iwslt", "{}_src_direction.txt".format(split))
    tgt_direction_path = os.path.join("directionFiles", "iwslt", "{}_tgt_direction.txt".format(split))
    if not os.path.exists(src_direction_path):
        raise ValueError(
            "{}_src_direction_path cannot be open, the current path is {}".format(split, src_direction_path))
    else:
        tmp_list_val = []
        with open(src_direction_path, 'r') as data:
            lines = data.readlines()
            for i in range(len(lines)):
                tmp_list_val.append(int(lines[i]))
                pass
            data.close()
        src_direction = torch.LongTensor(tmp_list_val)
        src_direction = RawLabelDataset(src_direction)
        pass
    if not os.path.exists(tgt_direction_path):
        raise ValueError(
            "{}_tgt_direction_path cannot be open, the current path is {}".format(split, tgt_direction_path))
    else:
        tmp_list_val = []
        with open(tgt_direction_path, 'r') as data:
            lines = data.readlines()
            for i in range(len(lines)):
                tmp_list_val.append(int(lines[i]))
                pass
            data.close()
        tgt_direction = torch.LongTensor(tmp_list_val)
        tgt_direction = RawLabelDataset(tgt_direction)
        pass

    return LanguagePairDatasetCllMerged(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        src_direction=src_direction,
        tgt_direction=tgt_direction,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@register_task("translation_cll_merged", dataclass=TranslationConfig)
class TranslationTaskCllMerged(TranslationTask):

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )
