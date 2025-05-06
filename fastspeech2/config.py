import os
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

    def __repr__(self):
        # with formatting
        return (
            "DatasetPathConfig( \n"
            f"    meta_file_train={self.meta_file_train}, \n"
            f"    meta_file_val={self.meta_file_val}, \n"
            f"    speaker_map_file={self.speaker_map_file}, \n"
            f"    feature_dir={self.feature_dir}, \n"
            f"    stats_file={self.stats_file}, \n"
            f"    sentiment_file={self.sentiment_file})"
        )
    
class DatasetPreprocessingConfig:
    def __init__(
        self,
        text_cleaners: list
    ):
        self.text_cleaners = text_cleaners
    
    def __repr__(self):
        return (
            "DatasetPreprocessingConfig( \n"
            f"    text_cleaners={self.text_cleaners})"
        )

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
    
    def __repr__(self):
        return (
            "DatasetFeaturePropertiesConfig( \n"
            f"    pitch_feature_level={self.pitch_feature_level}, \n"
            f"    energy_feature_level={self.energy_feature_level}, \n"
            f"    n_mel_channels={self.n_mel_channels}, \n"
            f"    max_wav_value={self.max_wav_value}, \n"
            f"    sampling_rate={self.sampling_rate}, \n"
            f"    stft_hop_length={self.stft_hop_length})"
        )
        

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

    def __repr__(self):
        return (
            "DatasetConfig( \n"
            f"  path_config={self.path_config}, \n"
            f"  feature_properties_config={self.feature_properties_config}, \n"
            f"  preprocessing_config={self.preprocessing_config})"
        )

    @classmethod
    def load_from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        
        config_dir = os.path.dirname(yaml_file)
        
        path_config = DatasetPathConfig(
            meta_file_train=os.path.join(config_dir, config['path']['meta_file_train']),
            meta_file_val=os.path.join(config_dir, config['path']['meta_file_val']),
            speaker_map_file=os.path.join(config_dir, config['path']['speaker_map_file']),
            feature_dir=os.path.join(config_dir, config['path']['feature_dir']),
            stats_file=os.path.join(config_dir, config['path']['stats_file']),
            sentiment_file=os.path.join(config_dir, config['path']['sentiment_file']) if 'sentiment_file' in config['path'] else None
        )
        
        feature_properties_config = DatasetFeaturePropertiesConfig(
            pitch_feature_level=config['properties']['pitch_feature_level'],
            energy_feature_level=config['properties']['energy_feature_level'],
            n_mel_channels=config['properties']['n_mel_channels'],
            max_wav_value=config['properties']['max_wav_value'],
            sampling_rate=config['properties']['sampling_rate'],
            stft_hop_length=config['properties']['stft_hop_length']
        )
        
        preprocessing_config = DatasetPreprocessingConfig(
            text_cleaners=config['preprocessing']['text_cleaners']
        )
        
        return cls(path_config, feature_properties_config, preprocessing_config)

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


    def __repr__(self):
        return (
            "ModelTransformerConfig( \n"
            f"    encoder_layer={self.encoder_layer}, \n"
            f"    encoder_head={self.encoder_head}, \n"
            f"    encoder_hidden={self.encoder_hidden}, \n"
            f"    decoder_layer={self.decoder_layer}, \n"
            f"    decoder_head={self.decoder_head}, \n"
            f"    decoder_hidden={self.decoder_hidden}, \n"
            f"    conv_filter_size={self.conv_filter_size}, \n"
            f"    conv_kernel_size={self.conv_kernel_size}, \n"
            f"    encoder_dropout={self.encoder_dropout}, \n"
            f"    decoder_dropout={self.decoder_dropout})"
        )

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
    
    def __repr__(self):
        return (
            "ModelVariancePredictorConfig( \n"
            f"    filter_size={self.filter_size}, \n"
            f"    kernel_size={self.kernel_size}, \n"
            f"    dropout={self.dropout})"
        )

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

    def __repr__(self):
        return (
            "ModelVarianceEmbeddingConfig( \n"
            f"    pitch_quantization={self.pitch_quantization}, \n"
            f"    energy_quantization={self.energy_quantization}, \n"
            f"    n_bins={self.n_bins})"
        )

class ModelVocoderConfig:
    def __init__(
        self,
        model: str,
        speaker: str,
    ):
        self.model = model
        self.speaker = speaker

    def __repr__(self):
        return (
            "ModelVocoderConfig( \n"
            f"    model={self.model}, \n"
            f"    speaker={self.speaker})"
        )

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

    def __repr__(self):
        return (
            "ModelGlobalConfig( \n"
            f"    multi_speaker={self.multi_speaker}, \n"
            f"    num_sentiments={self.num_sentiments}, \n"
            f"    max_seq_len={self.max_seq_len})"
        )

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

    def __repr__(self):
        return (
            "ModelConfig( \n"
            f"  transformer_config={self.transformer_config}, \n"
            f"  variance_predictor_config={self.variance_predictor_config}, \n"
            f"  variance_embedding_config={self.variance_embedding_config}, \n"
            f"  vocoder_config={self.vocoder_config}, \n"
            f"  global_config={self.global_config})"
        )

    @classmethod
    def load_from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        
        transformer_config = ModelTransformerConfig(
            encoder_layer=config['transformer']['encoder_layer'],
            encoder_head=config['transformer']['encoder_head'],
            encoder_hidden=config['transformer']['encoder_hidden'],
            decoder_layer=config['transformer']['decoder_layer'],
            decoder_head=config['transformer']['decoder_head'],
            decoder_hidden=config['transformer']['decoder_hidden'],
            conv_filter_size=config['transformer']['conv_filter_size'],
            conv_kernel_size=config['transformer']['conv_kernel_size'],
            encoder_dropout=config['transformer']['encoder_dropout'],
            decoder_dropout=config['transformer']['decoder_dropout']
        )
        
        variance_predictor_config = ModelVariancePredictorConfig(
            filter_size=config['variance_predictor']['filter_size'],
            kernel_size=config['variance_predictor']['kernel_size'],
            dropout=config['variance_predictor']['dropout']
        )
        
        variance_embedding_config = ModelVarianceEmbeddingConfig(
            pitch_quantization=config['variance_embedding']['pitch_quantization'],
            energy_quantization=config['variance_embedding']['energy_quantization'],
            n_bins=config['variance_embedding']['n_bins']
        )
        
        vocoder_config = ModelVocoderConfig(
            model=config['vocoder']['model'],
            speaker=config['vocoder']['speaker']
        )
        
        global_config = ModelGlobalConfig(
            multi_speaker=config['multi_speaker'],
            num_sentiments=config['num_sentiments'],
            max_seq_len=config['max_seq_len']
        )
        
        return cls(transformer_config, variance_predictor_config, variance_embedding_config, vocoder_config, global_config)

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

    def __repr__(self):
        return (
            "TrainStepConfig( \n"
            f"    total_step={self.total_step}, \n"
            f"    log_step={self.log_step}, \n"
            f"    synth_step={self.synth_step}, \n"
            f"    val_step={self.val_step}, \n"
            f"    save_step={self.save_step}, \n"
            f"    batch_size={self.batch_size})"
        )

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

    def __repr__(self):
        return (
            "TrainOptimizerConfig( \n"
            f"    betas={self.betas}, \n"
            f"    eps={self.eps}, \n"
            f"    weight_decay={self.weight_decay}, \n"
            f"    grad_clip_thresh={self.grad_clip_thresh}, \n"
            f"    grad_acc_step={self.grad_acc_step}, \n"
            f"    warm_up_step={self.warm_up_step}, \n"
            f"    anneal_steps={self.anneal_steps}, \n"
            f"    anneal_rate={self.anneal_rate})"
        )


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

    def __repr__(self):
        return (
            "TrainPathConfig( \n"
            f"    ckpt_path={self.ckpt_path}, \n"
            f"    log_path={self.log_path}, \n"
            f"    result_path={self.result_path})"
        )
    
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

    def __repr__(self):
        return (
            "TrainConfig( \n"
            f"  path_config={self.path_config}, \n"
            f"  optimizer_config={self.optimizer_config}, \n"
            f"  step_config={self.step_config})"
        )

    @classmethod
    def load_from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        config_dir = os.path.dirname(yaml_file)
        
        path_config = TrainPathConfig(
            ckpt_path=os.path.join(config_dir, config['path']['ckpt_path']),
            log_path=os.path.join(config_dir, config['path']['log_path']),
            result_path=os.path.join(config_dir, config['path']['result_path'])
        )
        
        optimizer_config = TrainOptimizerConfig(
            betas=config['optimizer']['betas'],
            eps=config['optimizer']['eps'],
            weight_decay=config['optimizer']['weight_decay'],
            grad_clip_thresh=config['optimizer']['grad_clip_thresh'],
            grad_acc_step=config['optimizer']['grad_acc_step'],
            warm_up_step=config['optimizer']['warm_up_step'],
            anneal_steps=config['optimizer']['anneal_steps'],
            anneal_rate=config['optimizer']['anneal_rate']
        )
        
        step_config = TrainStepConfig(
            total_step=config['step']['total_step'],
            log_step=config['step']['log_step'],
            synth_step=config['step']['synth_step'],
            val_step=config['step']['val_step'],
            save_step=config['step']['save_step'],
            batch_size=config['step']['batch_size']
        )
        
        return cls(path_config, optimizer_config, step_config)

def main():
    # Example usage
    dataset_config = DatasetConfig.load_from_yaml('config/LibriTTS/dataset.yaml')
    model_config = ModelConfig.load_from_yaml('config/LibriTTS/model.yaml')
    train_config = TrainConfig.load_from_yaml('config/LibriTTS/train.yaml')

    print(dataset_config)
    print(model_config)
    print(train_config)

if __name__ == "__main__":
    main()
