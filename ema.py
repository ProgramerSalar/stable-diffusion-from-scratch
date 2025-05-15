import torch 
from torch import nn 


class LitEma(nn.Module):


    def __init__(self,
                 model, 
                 decay=0.999,
                 use_num_updates=True):
        
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1.")
        

        # Dict mapping model parameter names to shadow param names 
        self.m_name2s_name = {}

        self.register_buffer(name='decay',
                             tensor=torch.tensor(decay, dtype=torch.float32))
        
        self.register_buffer(name='num_updates',
                             tensor=torch.tensor(0, dtype=torch.int) if use_num_updates \
                                else torch.tensor(-1, dtype=torch.int))
        
        # Initialize shadow param of all trainable param in the model 
        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove '.' char as they are not allowed in buffer names 
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(name=s_name, tensor=p.clone().detach().data)


        self.collected_parameters = []

    def reset_num_updates(self):

        """ Reset the number of updates counter to zero."""

        del self.num_updates
        self.register_buffer(name='num_updates',
                             tensor=torch.tensor(0, dtype=torch.int))
        

    def forward(self, 
                model):
        
        decay = self.decay 

        # Adjust decay based on number of updates if enabled 
        if self.num_updates >= 0:
            self.num_updates += 1 
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():

            # get current model param and shadow param 
            m_param = dict(model.named_parameters())
            shadow_param = dict(self.named_buffers())


            # update each shadow param 
            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]

                    # Ensure types match 
                    shadow_param[sname] = shadow_param[sname].type_as(m_param[key])
                    # EMA update: shadow = shadow - (1 - decay) * (shadow - current)
                    shadow_param[sname].sub(one_minus_decay * (shadow_param[sname] - m_param[key]))

                else:
                    assert not key in self.m_name2s_name


    def copy_to(self, 
                model):
        
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())

        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)

            else:
                assert not key in self.m_name2s_name



    def store(self,
              parameters):
        
        self.collected_parameters = [param.clone() for param in parameters]


    def restore(self, paramaters):

        for c_param, param in zip(self.collected_parameters, paramaters):
            param.data.copy_(c_param.data)







if __name__ == "__main__":

    from torch.utils.data import DataLoader, TensorDataset

    # Define a simple model 
    class SimpleModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
        

    # create a synthetic data 
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    # initialize model, optimizer, Ema 
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ema = LitEma(model, decay=0.999)


    # Training loop 
    num_epochs = 10 
    for epoch in range(num_epochs):

        model.train()
        for batch_x, batch_y in dataloader:

            # forward pass 
            outputs = model(batch_x)
            loss = nn.MSELoss()(outputs, batch_y)

            # backward pass and optimizers 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update Ema weights after each optimization step 
            ema(model)

        # validation phase (using Ema weights)
        model.eval()
        with torch.no_grad():

            # store original param 
            ema.store(model.parameters())

            # copy EMA weights to model 
            ema.copy_to(model)

            # validate with EMA weights 
            val_outputs = model(x)
            val_loss = nn.MSELoss()(val_outputs, y)
            print(f"Epoch {epoch + 1}, EMA validation Loss: {val_loss.item():.4f}")

            # Restore original param for continoue training 
            ema.restore(model.parameters())

        # save ema model checkpoint 
        if epoch % 5 == 0:
            ema.store(model.parameters())
            ema.copy_to(model)
            torch.save(model.state_dict(), f"ema_model_epoch{epoch}.pt")
            ema.restore(model.parameters())


    # final evaluation with EMA weights 
    ema.copy_to(model)
    test_loss = nn.MSELoss()(model(x), y)
    print(f"Final EMA Test loss: {test_loss.item():.4f}")




