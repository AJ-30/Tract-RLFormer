import numpy as np
import torch
import torch.nn as nn #ASHUTOSH

from tractRLformer.training.trainer import Trainer

#ASHUTOSH----
def normalize_vectors(v):
    return v / np.sqrt(np.sum(v ** 2, axis=-1, keepdims=True))

#Dynamic loss scaling ASHUTOSH
# class DynamicLossScaler:
#     def __init__(self, initial_scale=1.0, scale_factor=2.0, min_scale=1e-4, max_scale=1e5):
#         self.scale = initial_scale
#         self.scale_factor = scale_factor
#         self.min_scale = min_scale
#         self.max_scale = max_scale
#         self.moving_avg = None

#     def update_scale(self, loss):
#         if self.moving_avg is None:
#             self.moving_avg = loss
#         else:
#             self.moving_avg = 0.9 * self.moving_avg + 0.1 * loss  # Update moving average

#         if loss < 1e-6 * self.moving_avg:  # Loss is much smaller than average
#             self.scale /= self.scale_factor
#         elif loss > 100 * self.moving_avg:  # Loss is much larger than average
#             self.scale *= self.scale_factor

#         self.scale = max(self.min_scale, min(self.scale, self.max_scale))  # Bound the scale

#     def scale_loss(self, loss):
#         return loss * self.scale
#-----------
class SequenceTrainer(Trainer):
    # #------__init__ for learnable params for weighted loss function --------------------
    # def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, weight_fn1=1.0, weight_fn2=1.0):
    #     super(SequenceTrainer, self).__init__(model, optimizer, batch_size, get_batch, loss_fn, scheduler, eval_fns)
    #     self.weight_fn1 = nn.Parameter(torch.tensor(weight_fn1))
    #     self.weight_fn2 = nn.Parameter(torch.tensor(weight_fn2))
    # #-------------------------------------------------------------------------------------

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        # if torch.isnan(action_preds).any() or torch.isinf(action_predds).any():
        #     print("Input action_preds contains NaN or infinite values.")

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        # print('------------------------------------------')#ASHUTOSH
        # print(action_preds)
        # print(action_preds.shape)#[32x,3]
        # print(action_target)
        # print(action_target.shape)
        # print(type(action_preds[-1].detach().cpu().numpy()))
        # print(action_preds[-1].detach().cpu().numpy())
        # print(normalize_vectors(action_preds[-1].detach().cpu().numpy()).reshape(1,3).shape)
        # np.sum(normalize_vectors(action_preds[-1].detach().cpu().numpy()) * normalize_vectors(action_target[-1].detach().cpu().numpy()), axis=1)

        #LOSS FUNCTION:------------------------------------------------------------------------------------------ mod by AJ, ASHUTOSH
        # loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: 180/np.pi*(torch.acos(torch.clamp(torch.dot(a/torch.norm(a), a_hat/torch.norm(a_hat)), -1.0, 1.0)))
        # def loss_fn1(s_hat, a_hat, r_hat, s, a, r):
        #     loss = torch.zeros(a_hat.shape[0])  # Initialize loss tensor
        #     for i in range(len(a_hat)):
        #         if i < 2:  # Handle first two indices
        #             loss_i = 180/np.pi * (
        #                 torch.acos(torch.clamp(torch.dot(a[i]/torch.norm(a[i]), a_hat[i]/torch.norm(a_hat[i])), -1.0, 1.0)) +
        #                 torch.acos(torch.clamp(torch.dot(a[i+1]/torch.norm(a[i+1]), a_hat[i+1]/torch.norm(a_hat[i+1])), -1.0, 1.0))
        #             )
        #         elif i > len(a_hat) - 3:  # Handle last two indices
        #             loss_i = 180/np.pi * (
        #                 torch.acos(torch.clamp(torch.dot(a[i-1]/torch.norm(a[i-1]), a_hat[i-1]/torch.norm(a_hat[i-1])), -1.0, 1.0)) +
        #                 torch.acos(torch.clamp(torch.dot(a[i]/torch.norm(a[i]), a_hat[i]/torch.norm(a_hat[i])), -1.0, 1.0))
        #             )
        #         else:  # Regular cases
        #             loss_i = 180/np.pi * (
        #                 torch.acos(torch.clamp(torch.dot(a[i-2]/torch.norm(a[i-2]), a_hat[i-2]/torch.norm(a_hat[i-2])), -1.0, 1.0)) +
        #                 torch.acos(torch.clamp(torch.dot(a[i-1]/torch.norm(a[i-1]), a_hat[i-1]/torch.norm(a_hat[i-1])), -1.0, 1.0)) +
        #                 torch.acos(torch.clamp(torch.dot(a[i]/torch.norm(a[i]), a_hat[i]/torch.norm(a_hat[i])), -1.0, 1.0)) +
        #                 torch.acos(torch.clamp(torch.dot(a[i+1]/torch.norm(a[i+1]), a_hat[i+1]/torch.norm(a_hat[i+1])), -1.0, 1.0)) +
        #                 torch.acos(torch.clamp(torch.dot(a[i+2]/torch.norm(a[i+2]), a_hat[i+2]/torch.norm(a_hat[i+2])), -1.0, 1.0))
        #             )
        #         loss[i] = loss_i

        #     print(torch.sum(loss))

        #     return torch.sum(loss) #started from >4,10,000 nearly for fornix, 970 around actions in total
        #---------------------------------------------------------------------------------------------------------------------------------------------------------

        def loss_fn(s_hat, a_hat, r_hat, s, a, r):          #optimized
            epsilon = 1e-4  # Small constant to prevent division by zero

            loss = torch.zeros(a_hat.shape[0], device=states.device)  # Initialize loss tensor
            
            # Normalize vectors
            a_normalized = a / torch.norm(a, dim=1, keepdim=True).clamp(min=epsilon)
            a_hat_normalized = a_hat / torch.norm(a_hat, dim=1, keepdim=True).clamp(min=epsilon)
            
            # Calculate dot products
            dot_products = torch.einsum('bi,bi->b', [a_normalized, a_hat_normalized])

            # Calculate angles and don't convert to degrees 180 / 3.141592653589793 *
            angles =  torch.acos(torch.clamp(dot_products, -1.0 + epsilon, 1.0 - epsilon))

            # Fill loss tensor with angles, handling edge cases
            loss += angles

            loss += torch.cat([torch.tensor([0], device=states.device), angles[:-1]])
            loss += torch.cat([torch.tensor([0], device=states.device),torch.tensor([0], device=states.device), angles[:-2]])
            loss += torch.cat([angles[1:], torch.tensor([0], device=states.device)])
            loss += torch.cat([angles[2:], torch.tensor([0], device=states.device), torch.tensor([0], device=states.device)])

            return torch.sum(loss) #started from >4,10,000 nearly for fornix, 970 around actions in total
        #------------------------

        loss = loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        # #-----------------if weighted loss----------------------------------AS
        # loss_fn2 =lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2)
        # loss_fn2_value = loss_fn2(
        #     None, action_preds, None,
        #     None, action_target, None,
        # )

        # # Compute the weighted sum of the losses
        # weighted_loss = self.weight_fn1 * loss + self.weight_fn2 * loss_fn2_value
        # #-----------------if weighted loss----------------------------------AS


        # if torch.isnan(action_preds).any() or torch.isinf(action_preds).any():
        #     print("Input action_preds contains NaN or infinite values.")


        self.optimizer.zero_grad()
        loss.backward()
        # #-----------------if weighted loss----------------------------------AS
        # weighted_loss.backward()
        # #-----------------if weighted loss----------------------------------AS
        
        # from torch._six import inf
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25, error_if_nonfinite=False, norm_type=2.0)
        self.optimizer.step()
        # for param in self.model.parameters():
        #     print(f'\nRequire grad{param.requires_grad}\n\n')
        # print(list(self.model.parameters()))
        # for name, param in self.model.named_parameters():
        #     # print(f"Parameter name: {name}, Size: {param.grad}")
        #     if torch.isnan(param).any() or torch.isinf(param).any():
        #         print("---------------- param contains NaN or infinite values.")
        #         import sys
        #         sys.exit(1)

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
        # #-----------------if weighted loss----------------------------------AS
        # return weighted_loss.detach().cpu().item()
        # #-----------------if weighted loss----------------------------------AS
