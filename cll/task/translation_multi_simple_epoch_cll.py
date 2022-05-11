from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from cll.task.multilingual_data_manager_cll import MultilingualDatasetManagerCll


@register_task("translation_multi_simple_epoch_cll")
class TranslationMultiSimpleEpochTaskCLL(TranslationMultiSimpleEpochTask):

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.data_manager = MultilingualDatasetManagerCll.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )