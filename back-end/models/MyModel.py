



# bert + lstm
# epoch=10,num_warmup_step=3 * nbatch bert_lr=1e-5, other_lr=2e-5
# best result: {'val_loss': 0.6148558079250275, 'val_acc': 0.792, 'val_f1_score': 0.762089428579429}
# test_f1_score:  0.745
# test_acc:  0.776
# test_loss:  0.628


import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.num_labels = 6
        # BERT 'bert base chinese'
        self.bert = BertModel.from_pretrained('weights/bert-base-chinese', local_files_only=True)
        # 让参数可以更新
        for param in self.bert.parameters():
            param.requires_grad = True
        # LSTM
        self.lstm = nn.LSTM(768, 768, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.1)
        # 全连接层
        self.fc_rnn = nn.Linear(768 * 2, self.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            return_dict=False,
            output_hidden_states=True
    ):
        # bert.forward
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
            output_hidden_states=output_hidden_states
        )

        # encoder的输出
        encoder_out = outputs[0]
        # 第一个参数 是所有输入对应的输出  第二个参数 是 cls最后接的分类层的输出
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        # 只要序列中最后一个token对应的输出，（因为lstm会记录前边token的信息）
        # h_n：最后一个时间步的输出，即 h_n = output[:, -1, :]
        logits = self.fc_rnn(out[:, -1, :])

        # 如果是做预测就不用计算loss
        if labels == None:
            return SequenceClassifierOutput(
                logits=logits,
            )
        else:
            # train和eval，要计算loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
            )