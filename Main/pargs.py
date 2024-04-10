import argparse


def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_size', type=int, default=52000)
    parser.add_argument('--special_tokens', type=list,
                        default=["<@user>", "<url>"])

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--accumulation_batch_size', type=int, default=8064)
    parser.add_argument('--batch_size', type=int, default=72)

    parser.add_argument('--max_position_embeddings', type=int, default=128)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--type_vocab_size', type=int, default=2)

    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5")  # parallel

    parser.add_argument('--unsup_dataset', type=str, default='UTwitter')
    parser.add_argument('--post_num_limit', type=int, default=80)

    parser.add_argument('--pep_epochs', type=int, default=40)
    # parser.add_argument('--post_batch_size', type=int, default=32)
    parser.add_argument('--pep_batch_size', type=int, default=64)
    parser.add_argument('--b', type=int, default=20)

    parser.add_argument('--load_name', type=str, default='')

    args = parser.parse_args()
    return args
