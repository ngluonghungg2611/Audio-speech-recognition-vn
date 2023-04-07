from transformers import AutoTokenizer
import torch
from s2t_rec.punc_restore_vn.model import BertBLSTMPunc


class Inference:
    def __init__(self, weight_path, punc_path, device, tokenizer_pretrain):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_pretrain, do_lower_case=True
        )
        self.bert_blsmt_punc = BertBLSTMPunc(pretrained_token=tokenizer_pretrain).to(
            torch.device(device)
        )
        checkpoint = torch.load(weight_path, map_location=torch.device(device))
        # load model weights state_dict
        self.bert_blsmt_punc.load_state_dict(checkpoint)
        self.bert_blsmt_punc.eval()

        self.punc_list = []
        with open(punc_path, "r") as f:
            for line in f:
                self.punc_list.append(line.strip())
        self.punc_list = [0] + self.punc_list
        self.device = device

    def punc(self, text):
        """add punctuation to predict text

        Args:
            text (string): text of predict recognizer speech

        Returns:
            string: text after add punctuation
        """
        tokenized_input = self.tokenizer(text)
        input_ = tokenized_input["input_ids"][1:-1]
        input_ = torch.tensor(input_)
        input_ = input_.unsqueeze(0)
        input_ = input_.to(self.device)

        logits, _ = self.bert_blsmt_punc(input_)
        preds = torch.argmax(logits, dim=1).squeeze(0)

        tokens = self.tokenizer.convert_ids_to_tokens(
            tokenized_input["input_ids"][1:-1]
        )

        labels = preds.tolist()

        # add 0 for non punc
        text = ""
        for t, l in zip(tokens, labels):
            if t != "<unk>":
                text += t + " "
                if l != 0:  # Non punc.
                    text += self.punc_list[l] + " "
        return text
