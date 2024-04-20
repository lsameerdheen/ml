import torch
from transformers import pipeline,AutoConfig
#https://github.com/Data-drone/ANZ_LLM_Bootcamp/blob/master/2.0_Working_w_HuggingFace.py
torch.cuda.empty_cache()
run_mode = 'gpu'

if run_mode == 'cpu':
  from ctransformer import AutoModelforCasualLM,AutoTokenizer
  model_folder = 'C:/ai/hfmodels'
  model_id = 'Orca-2-13b'
  model = AutoModelForCausalLM.from_pretrained(f'{model_folder}/{model_id}',
                                              hf=True, local_files_only=True)
  tokenizer = AutoTokenizer.from_pretrained(model)
  pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer 
  )
elif run_mode == 'gpu':
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM, LlamaTokenizer 
  from accelerate import init_empty_weights,load_checkpoint_and_dispatch, infer_auto_device_map
  print(torch.cuda.memory_summary(device=None, abbreviated=False))
  model_folder = 'C:/ai/hfmodels'
  model_id = 'Orca-2-13b'
  #model_id = f''
  # note when on gpu then this will auto load to gpu
  # this will take approximately an extra 1GB of VRAM
  cached_model = f'{model_folder}/{model_id}' 
  tokenizer = AutoTokenizer.from_pretrained(cached_model, use_fast=False)  
  model = AutoModelForCausalLM.from_pretrained(cached_model,                                               
                                               device_map='cuda',
                                               from_tf=True
                                              )
 
  pipe = pipeline(
      "text-generation", model=model, tokenizer=tokenizer 
      )
  
def string_printer(out_obj, run_mode):
  #"""
  #Short convenience function because the output formats change between CPU and GPU
  #"""
  print(out_obj[0]['generated_text'])
# COMMAND ----------

# We seem to need to set the max length here for mpt model
#output = pipe("tell me about quran?", max_new_tokens=200, repetition_penalty=0.1)
#string_printer(output, run_mode)

# COMMAND ----------

# We seem to need to set the max length here for mpt model
#output = pipe("tell me about quran?", max_new_tokens=20, repetition_penalty=1.2)
#string_printer(output, run_mode)

# COMMAND ----------

# repetition_penalty affects whether we get repeats or not
#output = pipe("tell me about quran?", max_new_tokens=200, repetition_penalty=1.2)
#string_printer(output, run_mode)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Advanced Generation Config
# MAGIC For a full dive into generation config see the [docs](https://huggingface.co/docs/transformers/generation_strategies)\
# MAGIC **NOTE** `ctransformers` does not support all the same configs. See [docs](https://github.com/marella/ctransformers#method-llmgenerate)\
# MAGIC The ones that are supported will run the same way
# MAGIC **TODO** Need a better prompt to show off temperature / top_k

# COMMAND ----------

#output = pipe("Tell me about what makes a good burger?", max_new_tokens=200, repetition_penalty=1.2)
#string_printer(output, run_mode)

# COMMAND ----------

#output = pipe("Tell me about what makes a good burger?", max_new_tokens=200, repetition_penalty=1.2, top_k=100)
#string_printer(output, run_mode)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Picking a model
# MAGIC Whilst Model providers like Open-AI tend to have one generic model for all usecases, there is more nuance in OpenSource\
# MAGIC See: https://www.databricks.com/product/machine-learning/large-language-models-oss-guidance
# MAGIC Different OSS Models have different things that they are trained on.\
# MAGIC Lets look at the [MPT models](https://www.mosaicml.com/blog/mpt-7b) for example:
# MAGIC
# MAGIC This model comes in the variants:
# MAGIC - Base
# MAGIC - StoryWriter
# MAGIC - Instruct
# MAGIC - Chat
# MAGIC
# MAGIC `Base` is the common root for the models. The others are built on top of this.\
# MAGIC `Instruct` is built to follow instructions as per the [following paper](https://crfm.stanford.edu/2023/03/13/alpaca.html) \
# MAGIC At a high level we could say that OpenAI ChatGPT would more be a hybrid of Instruct and Chat rather than Base
# COMMAND ----------