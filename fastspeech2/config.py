import yaml


class DatasetPathConfig:
    def __init__(
        self,
        meta_file_train: str,
        meta_file_val: str,
        speaker_map_file: str,
        feature_dir: str,
        stats_file: str,
        sentiment_file: str = None,
    ):
        self.meta_file_train = meta_file_train
        self.meta_file_val = meta_file_val
        self.speaker_map_file = speaker_map_file
        self.feature_dir = feature_dir
        self.stats_file = stats_file
        self.sentiment_file = sentiment_file

class DatasetPreprocessingConfig:
    def __init__(
        self,
        text_cleaners: list
    ):
        self.text_cleaners = text_cleaners

class DatasetFeaturePropertiesConfig:
    def __init__(
        self,
        pitch_feature_level: str,
        energy_feature_level: str,
        n_mel_channels: int,
        max_wav_value: float,
        sampling_rate: int,
        stft_hop_length: int,
    ):
        self.pitch_feature_level = pitch_feature_level
        self.energy_feature_level = energy_feature_level
        self.n_mel_channels = n_mel_channels
        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.stft_hop_length = stft_hop_length
        

class DatasetConfig:
    def __init__(
        self,
        path_config: DatasetPathConfig,
        feature_properties_config: DatasetFeaturePropertiesConfig,
        preprocessing_config: DatasetPreprocessingConfig
    ):
        self.path_config = path_config
        self.feature_properties_config = feature_properties_config
        self.preprocessing_config = preprocessing_config

class ModelTransformerConfig:
    def __init__(
        self,
        encoder_layer: int,
        encoder_head: int,
        encoder_hidden: int,
        decoder_layer: int,
        decoder_head: int,
        decoder_hidden: int,
        conv_filter_size: int,
        conv_kernel_size: list,
        encoder_dropout: float,
        decoder_dropout: float
    ):
        self.encoder_layer = encoder_layer
        self.encoder_head = encoder_head
        self.encoder_hidden = encoder_hidden
        self.decoder_layer = decoder_layer
        self.decoder_head = decoder_head
        self.decoder_hidden = decoder_hidden
        self.conv_filter_size = conv_filter_size
        self.conv_kernel_size = conv_kernel_size
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

class ModelVariancePredictorConfig:
    def __init__(
        self,
        filter_size: int,
        kernel_size: int,
        dropout: float
    ):
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.dropout = dropout

class ModelVarianceEmbeddingConfig:
    def __init__(
        self,
        pitch_quantization: str,
        energy_quantization: str,
        n_bins: int
    ):
        self.pitch_quantization = pitch_quantization
        self.energy_quantization = energy_quantization
        self.n_bins = n_bins

class ModelVocoderConfig:
    def __init__(
        self,
        model: str,
        speaker: str,
    ):
        self.model = model
        self.speaker = speaker

class ModelGlobalConfig:
    def __init__(
        self,
        multi_speaker: bool,
        num_sentiments: int,
        max_seq_len: int
    ):
        self.multi_speaker = multi_speaker
        self.num_sentiments = num_sentiments
        self.max_seq_len = max_seq_len

class ModelConfig:

    def __init__(
        self,
        transformer_config: ModelTransformerConfig,
        variance_predictor_config: ModelVariancePredictorConfig,
        variance_embedding_config: ModelVarianceEmbeddingConfig,
        vocoder_config: ModelVocoderConfig,
        global_config: ModelGlobalConfig
    ):
        self.transformer_config = transformer_config
        self.variance_predictor_config = variance_predictor_config
        self.variance_embedding_config = variance_embedding_config
        self.vocoder_config = vocoder_config
        self.global_config = global_config


# path:
#   ckpt_path: "./output/ckpt/LibriTTS"
#   log_path: "./output/log/LibriTTS"
#   result_path: "./output/result/LibriTTS"
# optimizer:
#   batch_size: 16
#   betas: [0.9, 0.98]
#   eps: 0.000000001
#   weight_decay: 0.0
#   grad_clip_thresh: 1.0
#   grad_acc_step: 1
#   warm_up_step: 4000
#   anneal_steps: [300000, 400000, 500000]
#   anneal_rate: 0.3
# step:
#   total_step: 900000
#   log_step: 100
#   synth_step: 100
#   val_step: 100
#   # save_step: 100000
#   save_step: 100

class TrainStepConfig:
    def __init__(
        self,
        total_step: int,
        log_step: int,
        synth_step: int,
        val_step: int,
        save_step: int,
        batch_size: int
    ):
        self.total_step = total_step
        self.log_step = log_step
        self.synth_step = synth_step
        self.val_step = val_step
        self.save_step = save_step
        self.batch_size = batch_size

class TrainOptimizerConfig:
    def __init__(
        self,
        betas: list,
        eps: float,
        weight_decay: float,
        grad_clip_thresh: float,
        grad_acc_step: int,
        warm_up_step: int,
        anneal_steps: list,
        anneal_rate: float
    ):
        # self.batch_size = batch_size
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip_thresh = grad_clip_thresh
        self.grad_acc_step = grad_acc_step
        self.warm_up_step = warm_up_step
        self.anneal_steps = anneal_steps
        self.anneal_rate = anneal_rate


class TrainPathConfig:
    def __init__(
        self,
        ckpt_path: str,
        log_path: str,
        result_path: str
    ):
        self.ckpt_path = ckpt_path
        self.log_path = log_path
        self.result_path = result_path
    
class TrainConfig:
    def __init__(
        self,
        path_config: TrainPathConfig,
        optimizer_config: TrainOptimizerConfig,
        step_config: TrainStepConfig
    ):
        self.path_config = path_config
        self.optimizer_config = optimizer_config
        self.step_config = step_config