from transformers import pipeline, AutoConfig
import torch 

def loadllmaModule():
    if torch.cuda.is_available(): 
      dev = "cuda:1" 
    else: 
      dev = "cpu" 
    print(dev)
    device = torch.device(dev)

    run_mode= "gpu"
    if run_mode == 'cpu':
      ### Note that caching for TheBloke's models don't follow standard HuggingFace routine
      # You would need to `wget` then weights then use a model_path config instead.
      # See ctransformers docs for more info
      from ctransformers import AutoModelForCausalLM, AutoTokenizer
      model_id = 'Llama-2-7b-hf'
      bootcamp_dbfs_model_folder = 'D:/workspace/ameersoftware/learning/mldatasets/hugfaces/models'
      model = AutoModelForCausalLM.from_pretrained(f'{bootcamp_dbfs_model_folder}\{model_id}',
                                                  hf=True, local_files_only=True)
      tokenizer = AutoTokenizer.from_pretrained(model)

      pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer 
      )
      return pipe
    elif run_mode == 'gpu':
      from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM, LlamaTokenizer 
      from accelerate import init_empty_weights,load_checkpoint_and_dispatch, infer_auto_device_map
      # when loading from huggingface we need to set these
      model_id = 'Orca-2-13b'
      #model_revision = '40c5e2b32261834431f89850c8d5359631ffa764'
      bootcamp_dbfs_model_folder = 'c:/ai/hfmodels'
      dbfs_tmp_cache = 'c:/ai/hfmodel/tmpcache'
      # note when on gpu then this will auto load to gpu
      # this will take approximately an extra 1GB of VRAM
      cached_model = f'{bootcamp_dbfs_model_folder}/Orca-2-13b'  
      model_index_path  = 'c:/ai/hfmodels/Orca-2-13b'
      tokenizer = AutoTokenizer.from_pretrained(cached_model, cache_dir=dbfs_tmp_cache)
      model_config = AutoConfig.from_pretrained(cached_model)   
        # device_map = `auto` moves the model to GPU if possible.
      # Note not all models support `auto`
        #model = LlamaForCausalLM.from_pretrained(cached_model,
      #                                              config=model_config,
      #                                              device_map='cuda',
      #                                              torch_dtype=torch.bfloat16, # This will only work A10G / A100 and newer GPUs
      #                                              cache_dir=dbfs_tmp_cache
      #                                              )
      #with Python acceleration package improving the performance
      with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(cached_model,
                                                    config=model_config,
                                                    low_cpu_mem_usage=True
                                                    ).half()
      model = model.to(device)
      device_map = infer_auto_device_map(
          model,
          max_memory={0: "12000MiB"},
      ) 


      model = load_checkpoint_and_dispatch(
        model,
        model_index_path,
        device_map=device_map,
        no_split_module_classes=["LlamaDecoderLayer"],
        dtype=torch.float16,
        offload_folder="offload",
        offload_state_dict=True
      )   
      
    pipe = pipeline(
     "text-generation", model=model, tokenizer=tokenizer)
    return pipe
pipe =  loadllmaModule()     
    
  
def string_printer(out_obj, run_mode):
#  """
#  Short convenience function because the output formats change between CPU and GPU
#  """
  result = out_obj[0]['generated_text']
  print(result)
  return result
## We seem to need to set the max length here for mpt model
output = pipe("Tell me about quran?", max_new_tokens=200, repetition_penalty=0.1)
string_printer(output, run_mode)
#
## COMMAND ----------
#
## We seem to need to set the max length here for mpt model
#output = pipe("Tell me how you have been and any signifcant things that have happened to you?", max_new_tokens=20, repetition_penalty=1.2)
#string_printer(output, run_mode)
#cls
## COMMAND ----------
#
## repetition_penalty affects whether we get repeats or not
#output = pipe("Tell me how you have been and any signifcant things that have happened to you?", max_new_tokens=200, repetition_penalty=1.2)
#string_printer(output, run_mode)
#
## COMMAND ----------
#
## MAGIC %md
## MAGIC ### Advanced Generation Config
## MAGIC For a full dive into generation config see the [docs](https://huggingface.co/docs/transformers/generation_strategies)\
## MAGIC **NOTE** `ctransformers` does not support all the same configs. See [docs](https://github.com/marella/ctransformers#method-llmgenerate)\
## MAGIC The ones that are supported will run the same way
## MAGIC **TODO** Need a better prompt to show off temperature / top_k
#
## COMMAND ----------
#
#output = pipe("Tell me about what makes a good burger?", max_new_tokens=200, repetition_penalty=1.2)
#string_printer(output, run_mode)
#
## COMMAND ----------
#
#output = pipe("Tell me about what makes a good burger?", max_new_tokens=200, repetition_penalty=1.2, top_k=100)
#string_printer(output, run_mode)