import copy
import logging

from transformers import LEDConfig, LEDForConditionalGeneration, BertTokenizer, BartConfig
from transformers import BartForConditionalGeneration
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger("longformer-chinese")
logging.basicConfig(level=logging.INFO)

max_encoder_position_embeddings = 8192
max_decoder_position_embeddings = 1024

# led_config = BartConfig.from_json_file(r'E:\chz1\pythonRep\transformers\src\transformers\models\led\model\fnlp-bart\config.json')
led_config = LEDConfig(
        vocab_size=51271,
        max_encoder_position_embeddings=max_encoder_position_embeddings,
        max_decoder_position_embeddings=max_decoder_position_embeddings,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        init_std=0.02,
        decoder_start_token_id=102,
        classifier_dropout=0.0,
        pad_token_id=0,
        bos_token_id=101,
        eos_token_id=102,
        attention_window=512
)
led_model = LEDForConditionalGeneration(led_config)

model_name = r"E:\chz1\pythonRep\transformers\src\transformers\models\led\model\fnlp-bart"
# bart_model = BartForConditionalGeneration.from_pretrained(model_name)
# tokenizer = BertTokenizer.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
bart_model = BartForConditionalGeneration.from_pretrained(model_name)


current_max_pos, embed_size = bart_model.model.encoder.embed_positions.weight.shape
new_encoder_pos_embed = bart_model.model.encoder.embed_positions.weight.new_empty(max_encoder_position_embeddings, embed_size)

k = 0
step = current_max_pos - 2
# new_encoder_pos_embed[0]=bart_model.model.encoder.embed_positions.weight[0]
encoder_position_embeddings = bart_model.model.encoder.embed_positions.weight[2:]
while k < max_encoder_position_embeddings:
    new_encoder_pos_embed[k:(k + step)] = encoder_position_embeddings
    k += step
led_model.base_model.encoder.embed_positions.weight.data = new_encoder_pos_embed

current_max_pos, embed_size = bart_model.model.decoder.embed_positions.weight.shape
new_decoder_pos_embed = bart_model.model.decoder.embed_positions.weight.new_empty(max_decoder_position_embeddings, embed_size)

k = 0
step = current_max_pos - 2
# new_encoder_pos_embed[0]=bart_model.model.encoder.embed_positions.weight[0]
decoder_position_embeddings = bart_model.model.decoder.embed_positions.weight[2:]
while k < max_decoder_position_embeddings:
    new_decoder_pos_embed[k:(k + step)] = decoder_position_embeddings
    k += step
led_model.base_model.decoder.embed_positions.weight.data = new_decoder_pos_embed

for i, (bart_encoder_layer, led_encoder_layer) in enumerate(
        zip(bart_model.model.encoder.layers, led_model.base_model.encoder.layers)):
    led_encoder_layer.self_attn.longformer_self_attn.key = bart_encoder_layer.self_attn.k_proj
    led_encoder_layer.self_attn.longformer_self_attn.query = bart_encoder_layer.self_attn.q_proj
    led_encoder_layer.self_attn.longformer_self_attn.value = bart_encoder_layer.self_attn.v_proj
    led_encoder_layer.self_attn.longformer_self_attn.key_global = copy.deepcopy(bart_encoder_layer.self_attn.k_proj)
    led_encoder_layer.self_attn.longformer_self_attn.query_global = copy.deepcopy(bart_encoder_layer.self_attn.q_proj)
    led_encoder_layer.self_attn.longformer_self_attn.value_global = copy.deepcopy(bart_encoder_layer.self_attn.v_proj)
    led_encoder_layer.self_attn.output = bart_encoder_layer.self_attn.out_proj
    led_encoder_layer.self_attn_layer_norm = bart_encoder_layer.self_attn_layer_norm
    led_encoder_layer.fc1 = bart_encoder_layer.fc1
    led_encoder_layer.fc2 = bart_encoder_layer.fc2
    led_encoder_layer.final_layer_norm = bart_encoder_layer.final_layer_norm

for i, (bart_decoder_layer, led_decoder_layer) in enumerate(
        zip(bart_model.model.decoder.layers, led_model.base_model.decoder.layers)):
    led_decoder_layer.self_attn.k_proj = bart_decoder_layer.self_attn.k_proj
    led_decoder_layer.self_attn.q_proj = bart_decoder_layer.self_attn.q_proj
    led_decoder_layer.self_attn.v_proj = bart_decoder_layer.self_attn.v_proj
    led_decoder_layer.self_attn.out_proj = bart_decoder_layer.self_attn.out_proj
    led_decoder_layer.self_attn_layer_norm = bart_decoder_layer.self_attn_layer_norm
    led_decoder_layer.encoder_attn.k_proj = bart_decoder_layer.encoder_attn.k_proj
    led_decoder_layer.encoder_attn.q_proj = bart_decoder_layer.encoder_attn.q_proj
    led_decoder_layer.encoder_attn.v_proj = bart_decoder_layer.encoder_attn.v_proj
    led_decoder_layer.encoder_attn_layer_norm = bart_decoder_layer.encoder_attn_layer_norm

    led_decoder_layer.fc1 = bart_decoder_layer.fc1
    led_decoder_layer.fc2 = bart_decoder_layer.fc2
    led_decoder_layer.final_layer_norm = bart_decoder_layer.final_layer_norm

led_model.lm_head = bart_model.lm_head

logger.info("convert bart-chinese to led success")
led_model.save_pretrained(r'E:\chz1\pythonRep\transformers\src\transformers\models\led\model\led_init')
tokenizer.save_pretrained(r'E:\chz1\pythonRep\transformers\src\transformers\models\led\model\led_init')
