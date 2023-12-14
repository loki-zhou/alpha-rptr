import torch
import torch.nn.functional as F

# action_logits = torch.rand(5)
# action_probs = F.softmax(action_logits, dim=-1)
action_probs = torch.FloatTensor([2, 2])

dist = torch.distributions.Categorical(action_probs)
action = dist.sample()
print(dist.log_prob(action), torch.log(action_probs[action]))