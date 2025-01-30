import os
from FTaaS.inter.job import Job

if __name__ == '__main__':
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".")) + '/main.py'
    job = Job()
    job.run('10.14.5.12', 12345, script_path, 
            ["--model_name_or_path google-bert/bert-base-uncased",
            "--dataset_name squad",
            "--do_train",
            "--do_eval",
            "--per_device_train_batch_size 16",
            "--learning_rate 1e-8",
            "--num_train_epochs 2",
            "--max_seq_length 384",
            "--evaluation_strategy steps",
            "--eval_steps 500",
            "--doc_stride 128",
            "--output_dir ./tmp_0/run",
            "--logging_steps 100",
            "--gradient_accumulation_steps 2",
            "--save_steps 200000"],
            ["export LD_LIBRARY_PATH='/home/cchen/miniconda3/envs/wgj/lib'"])
    
    
    