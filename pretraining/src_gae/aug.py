import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size*2 if args.conv == 'both' else args.hidden_size
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, 1))
        self.eps = 1e-15

    def forward(self, emb):
        return self.cls(emb)

    def dc_loss(self, emb):
        real = torch.randn_like(emb)
        out_real = self.forward(real)
        out_fake = self.forward(emb)
        #print("L26 embed size", out_real.size(), out_fake.size())
        pos_loss = -torch.log(out_real + self.eps).mean()
        neg_loss = -torch.log(1 - out_fake + self.eps).mean()
        return pos_loss + neg_loss
    
    def gen_loss(self, emb):
        out = self.forward(emb)
        return -torch.log(out + self.eps).mean()


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embed'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                print(name)
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embed'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32),
             reduction=reduction)
    return loss


def adv_project(grad, norm_type='inf', eps=1e-6):
    if norm_type == 'l2':
        direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
    elif norm_type == 'l1':
        direction = grad.sign()
    else:
        direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
    return direction

def get_newembed(args, embed):
    noise = embed.data.new(embed.size()).normal_(0, 1) * args.noise_var
    noise.requires_grad_()
    newembed = embed.data.detach() + noise
    return newembed, noise

def get_gradaug_newembed(args, embed, noise, adv_loss):
    delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=True)
    norm = delta_grad.norm()
    if (torch.isnan(norm) or torch.isinf(norm)):
        return {"success": False}
    # line 6 inner sum
    noise = noise + delta_grad * args.adv_step_size
    # line 6 projection
    noise = adv_project(noise, norm_type=args.project_norm_type, eps=args.noise_gamma)
    newembed = embed.data.detach() + noise
    newembed = newembed.detach()
    return {"success": True, "embed":embed}

def alum_loss(args, loss, embed, model, batch, logits):
    embed1, embed2 = embed
    newembed1, noise1 = get_newembed(args, embed1) 
    newembed2, noise2 = get_newembed(args, embed1) 

    ret = model(**batch, embed=(newembed1, newembed2))
    adv_logits = ret["sent_scores"]
    adv_loss = KL(adv_logits, logits.detach(), reduction="batchmean")
    # line 5, g_adv
    ret1 = get_gradaug_newembed(args, embed1, noise1, adv_loss)
    ret2 = get_gradaug_newembed(args, embed2, noise2, adv_loss)
    if not (ret1["success"] and ret2["success"]):
        return loss
    ret = model(**batch, embed=(ret1["embed"], ret2["embed"]))
    adv_logits = ret["sent_scores"]
    # line 8 symmetric KL
    adv_loss_f = KL(adv_logits, logits.detach())
    adv_loss_b = KL(logits, adv_logits.detach())
    adv_loss = (adv_loss_f + adv_loss_b) * args.adv_alpha
    loss = loss + adv_loss
    return loss