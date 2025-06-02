import os
import yaml

class DatasetPathConfig:
    def __init__(
        self,
        base_dir: str,
        meta_file_train: str,
        meta_file_val: str,
        speaker_map_file: str,
        feature_dir: str,
        stats_file: str,
        sentiment_file: str | None = None,
    ):
        self.meta_file_train = os.path.join(base_dir, meta_file_train)
        self.meta_file_val = os.path.join(base_dir, meta_file_val)
        self.speaker_map_file = os.path.join(base_dir, speaker_map_file)
        self.feature_dir = os.path.join(base_dir, feature_dir)
        self.stats_file = os.path.join(base_dir, stats_file)
        self.sentiment_file = os.path.join(base_dir, sentiment_file) if sentiment_file else None

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

    def to_dict(self):
        return {
            "meta_file_train": self.meta_file_train,
            "meta_file_val": self.meta_file_val,
            "speaker_map_file": self.speaker_map_file,
            "feature_dir": self.feature_dir,
            "stats_file": self.stats_file,
            "sentiment_file": self.sentiment_file
        }
    
    @classmethod
    def from_dict(cls, base_dir: str, config_dict: dict):
        return cls(
            base_dir=base_dir,
            meta_file_train=config_dict['meta_file_train'],
            meta_file_val=config_dict['meta_file_val'],
            speaker_map_file=config_dict['speaker_map_file'],
            feature_dir=config_dict['feature_dir'],
            stats_file=config_dict['stats_file'],
            sentiment_file=config_dict.get('sentiment_file', None)
        )
    
class DatasetPreprocessingConfig:
    def __init__(
        self,
        lexicon_path: str,
        text_cleaners: list,
    ):
        self.lexicon_path = lexicon_path
        self.text_cleaners = text_cleaners
    
    def __repr__(self):
        return (
            "DatasetPreprocessingConfig( \n"
            f"    lexicon_path={self.lexicon_path}, \n"
            f"    text_cleaners={self.text_cleaners})"
        )

    def to_dict(self):
        return {
            "lexicon_path": self.lexicon_path,
            "text_cleaners": self.text_cleaners
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            lexicon_path=config_dict['lexicon_path'],
            text_cleaners=config_dict['text_cleaners']
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
        num_sentiments: int,
    ):
        self.pitch_feature_level = pitch_feature_level
        self.energy_feature_level = energy_feature_level
        self.n_mel_channels = n_mel_channels
        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.stft_hop_length = stft_hop_length
        self.num_sentiments = num_sentiments
    
    def __repr__(self):
        return (
            "DatasetFeaturePropertiesConfig( \n"
            f"    pitch_feature_level={self.pitch_feature_level}, \n"
            f"    energy_feature_level={self.energy_feature_level}, \n"
            f"    n_mel_channels={self.n_mel_channels}, \n"
            f"    max_wav_value={self.max_wav_value}, \n"
            f"    sampling_rate={self.sampling_rate}, \n"
            f"    stft_hop_length={self.stft_hop_length}), \n"
            f"    num_sentiments={self.num_sentiments}"
        )

    def to_dict(self):
        return {
            "pitch_feature_level": self.pitch_feature_level,
            "energy_feature_level": self.energy_feature_level,
            "n_mel_channels": self.n_mel_channels,
            "max_wav_value": self.max_wav_value,
            "sampling_rate": self.sampling_rate,
            "stft_hop_length": self.stft_hop_length,
            "num_sentiments": self.num_sentiments
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            pitch_feature_level=config_dict['pitch_feature_level'],
            energy_feature_level=config_dict['energy_feature_level'],
            n_mel_channels=config_dict['n_mel_channels'],
            max_wav_value=config_dict['max_wav_value'],
            sampling_rate=config_dict['sampling_rate'],
            stft_hop_length=config_dict['stft_hop_length'],
            num_sentiments=config_dict['num_sentiments']
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
            f"  path={self.path_config}, \n"
            f"  properties={self.feature_properties_config}, \n"
            f"  preprocessing={self.preprocessing_config})"
        )

    def to_dict(self):
        return {
            "path": self.path_config.to_dict(),
            "properties": self.feature_properties_config.to_dict(),
            "preprocessing": self.preprocessing_config.to_dict()
        }
    
    @classmethod
    def from_dict(cls, base_dir: str, config_dict: dict):
        path_config = DatasetPathConfig.from_dict(base_dir, config_dict['path'])
        feature_properties_config = DatasetFeaturePropertiesConfig.from_dict(config_dict['properties'])
        preprocessing_config = DatasetPreprocessingConfig.from_dict(config_dict['preprocessing'])
        
        return cls(path_config, feature_properties_config, preprocessing_config)

    @classmethod
    def load_from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r', encoding="utf-8") as file:
            config = yaml.safe_load(file)
        
        config_dir = os.path.dirname(yaml_file)
        
        return cls.from_dict(
            base_dir=config_dir,
            config_dict=config
        )

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

    def to_dict(self):
        return {
            "encoder_layer": self.encoder_layer,
            "encoder_head": self.encoder_head,
            "encoder_hidden": self.encoder_hidden,
            "decoder_layer": self.decoder_layer,
            "decoder_head": self.decoder_head,
            "decoder_hidden": self.decoder_hidden,
            "conv_filter_size": self.conv_filter_size,
            "conv_kernel_size": self.conv_kernel_size,
            "encoder_dropout": self.encoder_dropout,
            "decoder_dropout": self.decoder_dropout
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            encoder_layer=config_dict['encoder_layer'],
            encoder_head=config_dict['encoder_head'],
            encoder_hidden=config_dict['encoder_hidden'],
            decoder_layer=config_dict['decoder_layer'],
            decoder_head=config_dict['decoder_head'],
            decoder_hidden=config_dict['decoder_hidden'],
            conv_filter_size=config_dict['conv_filter_size'],
            conv_kernel_size=config_dict['conv_kernel_size'],
            encoder_dropout=config_dict['encoder_dropout'],
            decoder_dropout=config_dict['decoder_dropout']
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
    
    def to_dict(self):
        return {
            "filter_size": self.filter_size,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            filter_size=config_dict['filter_size'],
            kernel_size=config_dict['kernel_size'],
            dropout=config_dict['dropout']
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
    
    def to_dict(self):
        return {
            "pitch_quantization": self.pitch_quantization,
            "energy_quantization": self.energy_quantization,
            "n_bins": self.n_bins
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            pitch_quantization=config_dict['pitch_quantization'],
            energy_quantization=config_dict['energy_quantization'],
            n_bins=config_dict['n_bins']
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
    
    def to_dict(self):
        return {
            "model": self.model,
            "speaker": self.speaker
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            model=config_dict['model'],
            speaker=config_dict['speaker']
        )

class ModelGlobalConfig:
    def __init__(
        self,
        multi_speaker: bool,
        use_sentiment: bool,
        max_seq_len: int
    ):
        self.multi_speaker = multi_speaker
        self.use_sentiment = use_sentiment
        self.max_seq_len = max_seq_len

    def __repr__(self):
        return (
            "ModelGlobalConfig( \n"
            f"    multi_speaker={self.multi_speaker}, \n"
            f"    use_sentiment={self.use_sentiment}, \n"
            f"    max_seq_len={self.max_seq_len})"
        )
    
    def to_dict(self):
        return {
            "multi_speaker": self.multi_speaker,
            "use_sentiment": self.use_sentiment,
            "max_seq_len": self.max_seq_len
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            multi_speaker=config_dict['multi_speaker'],
            use_sentiment=config_dict['use_sentiment'],
            max_seq_len=config_dict['max_seq_len']
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
    
    def to_dict(self):
        return {
            "transformer_config": self.transformer_config.to_dict(),
            "variance_predictor_config": self.variance_predictor_config.to_dict(),
            "variance_embedding_config": self.variance_embedding_config.to_dict(),
            "vocoder_config": self.vocoder_config.to_dict(),
            "global_config": self.global_config.to_dict()
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        transformer_config = ModelTransformerConfig.from_dict(config_dict['transformer'])
        variance_predictor_config = ModelVariancePredictorConfig.from_dict(config_dict['variance_predictor'])
        variance_embedding_config = ModelVarianceEmbeddingConfig.from_dict(config_dict['variance_embedding'])
        vocoder_config = ModelVocoderConfig.from_dict(config_dict['vocoder'])
        global_config = ModelGlobalConfig.from_dict(config_dict)
        
        return cls(transformer_config, variance_predictor_config, variance_embedding_config, vocoder_config, global_config)

    @classmethod
    def load_from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config)
class TrainOutputConfig:
    def __init__(
        self,
        log_dir_name: str,
        ckpt_dir_name: str,
    ):
        self.log_dir_name = log_dir_name
        self.ckpt_dir_name = ckpt_dir_name

    def __repr__(self):
        return (
            "TrainOutputConfig( \n"
            f"    log_dir_name={self.log_dir_name}, \n"
            f"    ckpt_dir_name={self.ckpt_dir_name})"
        )

    def to_dict(self):
        return {
            "log_dir_name": self.log_dir_name,
            "ckpt_dir_name": self.ckpt_dir_name
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            log_dir_name=config_dict['log_dir_name'],
            ckpt_dir_name=config_dict['ckpt_dir_name']
        )

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

    def to_dict(self):
        return {
            "total_step": self.total_step,
            "log_step": self.log_step,
            "synth_step": self.synth_step,
            "val_step": self.val_step,
            "save_step": self.save_step,
            "batch_size": self.batch_size
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            total_step=config_dict['total_step'],
            log_step=config_dict['log_step'],
            synth_step=config_dict['synth_step'],
            val_step=config_dict['val_step'],
            save_step=config_dict['save_step'],
            batch_size=config_dict['batch_size']
        )

class TrainOptimizerConfig:
    def __init__(
        self,
        init_lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        grad_clip_thresh: float,
        grad_acc_step: int,
        warm_up_step: int,
        anneal_steps: list,
        anneal_rate: float
    ):
        # self.batch_size = batch_size
        self.init_lr = init_lr
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
            f"    init_lr={self.init_lr}, \n"
            f"    betas={self.betas}, \n"
            f"    eps={self.eps}, \n"
            f"    weight_decay={self.weight_decay}, \n"
            f"    grad_clip_thresh={self.grad_clip_thresh}, \n"
            f"    grad_acc_step={self.grad_acc_step}, \n"
            f"    warm_up_step={self.warm_up_step}, \n"
            f"    anneal_steps={self.anneal_steps}, \n"
            f"    anneal_rate={self.anneal_rate})"
        )
    
    def to_dict(self):
        return {
            "init_lr": self.init_lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "grad_clip_thresh": self.grad_clip_thresh,
            "grad_acc_step": self.grad_acc_step,
            "warm_up_step": self.warm_up_step,
            "anneal_steps": self.anneal_steps,
            "anneal_rate": self.anneal_rate
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            init_lr=config_dict['init_lr'],
            betas=config_dict['betas'],
            eps=config_dict['eps'],
            weight_decay=config_dict['weight_decay'],
            grad_clip_thresh=config_dict['grad_clip_thresh'],
            grad_acc_step=config_dict['grad_acc_step'],
            warm_up_step=config_dict['warm_up_step'],
            anneal_steps=config_dict['anneal_steps'],
            anneal_rate=config_dict['anneal_rate']
        )
    
class TrainConfig:
    def __init__(
        self,
        output_config: TrainOutputConfig,
        optimizer_config: TrainOptimizerConfig,
        step_config: TrainStepConfig
    ):
        self.output_config = output_config
        self.optimizer_config = optimizer_config
        self.step_config = step_config

    def __repr__(self):
        return (
            "TrainConfig( \n"
            f"  output_config={self.output_config}, \n"
            f"  optimizer_config={self.optimizer_config}, \n"
            f"  step_config={self.step_config})"
        )

    def to_dict(self):
        return {
            "output": self.output_config.to_dict(),
            "optimizer_config": self.optimizer_config.to_dict(),
            "step_config": self.step_config.to_dict()
        }


    @classmethod
    def from_dict(cls, config_dict: dict):
        output_config = TrainOutputConfig.from_dict(config_dict['output'])
        optimizer_config = TrainOptimizerConfig.from_dict(config_dict['optimizer'])
        step_config = TrainStepConfig.from_dict(config_dict['step'])
        
        return cls(output_config, optimizer_config, step_config)

    @classmethod
    def load_from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config)

def main():
    # Example usage
    dataset_config = DatasetConfig.load_from_yaml('config/LibriTTS/dataset.yaml')
    model_config = ModelConfig.load_from_yaml('config/LibriTTS/model.yaml')
    train_config = TrainConfig.load_from_yaml('config/LibriTTS/train.yaml')

    print(dataset_config)
    print(model_config)
    print(train_config)

    # dump to dict
    dataset_config_dict = dataset_config.to_dict()
    model_config_dict = model_config.to_dict()
    train_config_dict = train_config.to_dict()
    print(dataset_config_dict)
    print(model_config_dict)
    print(train_config_dict)

if __name__ == "__main__":
    main()
