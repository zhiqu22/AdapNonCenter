from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from sd.task.multilingual_data_manager_sd import MultilingualDatasetManagerSd


@register_task("translation_multi_simple_epoch_sd")
class TranslationMultiSimpleEpochTaskSd(TranslationMultiSimpleEpochTask):

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.data_manager = MultilingualDatasetManagerSd.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )