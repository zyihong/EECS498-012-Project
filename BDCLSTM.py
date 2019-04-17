
# coding: utf-8

# In[2]:


import torch.nn as nn
from torch.autograd import Variable
import torch
from data.dataloader import load_data, FacadeDataset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim


# In[3]:

#test BDCLSTM
batch_size = 1
T = 20
H,W = 240,320
input_dim = 1
hidden_dim = 1
EPOCH = 10
PRINT_EVERY = 2
LEARNING_RATE = 1e-2


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        #return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
        #        Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))

class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=True):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        hidden_state = self._init_hidden(input_tensor.size(0),hidden_state)

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
                
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size,hidden_state):
        init_states = []
        if type(hidden_state) == None:
            for i in range(self.num_layers):
                init_states.append(self.cell_list[i].init_hidden(batch_size))
        #implement initialization with hidden state
        else:
            if not len(hidden_state)==self.num_layers:
                raise ValueError('Inconsistent list length.')
            for i in range(self.num_layers):
                init_states.append(hidden_state[i])
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# In[23]:


class BDClstm(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size=(3,3), num_layers=1,
                 batch_first=True, bias=True, return_all_layers=True):
        super(BDClstm, self).__init__()
        
        """
        input_size: (int,int) h,w of segmentation image
        input_dim: # channel of segmentation image
        hidden_dim: # channel of hidden state(output dim)
        kernel_size: default to be (3,3)
        num_layers: default to be 1
        
        """
        
        self.forward_lstm = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers)
        self.backward_lstm = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers)
        
        self.conv = nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1)
                              
    def forward(self, T, input_tensor, forward_initial_states,backward_initial_states,train=True):
        """
        T : length of sequence
        input_tensor: segmentation image of size (Batch_size,T,input_dim,h,w)
        forward_start_state: list of tuple of (initial hidden state(Batch_size,hidden_dim,h,w), initial_cell_state(zeros, same size as initial hidden state))
        backward_start_state: replace forward_start_state with backward(initial hidden state, initial cell state)
        
        output:
        h_out: hidden state output(N,T,hidden_dim,H,W)
        forward_c_out,backward_c_out : (N, hidden_dim, H,W)
        """
        
            
        (N,hidden_dim,H,W) = forward_initial_states[0][0].size()
        
        if not train:
            input_tensor = torch.zeros(N,T,hidden_dim,H,W)
                              
        forward_h_out,forward_c_out = self.forward_lstm(input_tensor, forward_initial_states)
        forward_h_out = forward_h_out[0]
        forward_c_out = forward_c_out[0]
        #invert input_tensor along time step
        idx = [i for i in range(input_tensor.size(1)-1,-1,-1)]
        # print(len(idx))
        idx = torch.LongTensor(idx)
        inverted_tensor = input_tensor.index_select(1,idx)
        
        backward_h_out,backward_c_out = self.backward_lstm(inverted_tensor, backward_initial_states)
        backward_h_out = backward_h_out[0]
        backward_c_out = backward_c_out[0]
        # print("backward_h_out size", backward_h_out.size())
        #invert output back to T=0,1,2,3...
        inverted_h_out = backward_h_out.index_select(1,idx)
        
        #concatenate forward and backward hidden state output along channel
        h_out_cat = torch.cat((forward_h_out,inverted_h_out), dim=2)
        
        h_out_cat = h_out_cat.view(-1,2*hidden_dim,H,W)
        h_out = self.conv(h_out_cat)
        #(N*T,hidden_dim,H,W)
        
        h_out = h_out.view(N,-1,hidden_dim,H,W)
        
        return h_out,forward_c_out,backward_c_out
        
                              
                              
        
        


# In[25]:




'''
Example:
input_tensor: size (N,T,input_dim,H,W) segmentation map input at each time step. During testing it should be None
forward_initial_states: list of (initial h, initial c), default to be length 1

backward_initial_states: list of (initial backward h, initial backward c), default to be length 1


'''


def split_T(input):
    num = input.shape[1]
    shuffle = np.arange(num)
    np.random.shuffle(shuffle)

    data_set = []
    it_nums = int(np.ceil(num / T))

    for i in range(it_nums):
        data_set.append((input[:, shuffle[i * T:(i + 1) * T], :, :, :], input[:, shuffle[i * T:(i + 1) * T], :, :, :]))

    return data_set


def test(testloader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            test_data, labels = data
            test_data = test_data.to(device)
            labels = labels.to(device)

            forward_initial_states = []
            forward_initial_state = (
            torch.rand(batch_size, hidden_dim, H, W).to(device), torch.zeros(batch_size, hidden_dim, H, W).to(device))
            forward_initial_states.append(forward_initial_state)
            backward_initial_states = []
            backward_initial_state = (
            torch.rand(batch_size, hidden_dim, H, W).to(device), torch.zeros(batch_size, hidden_dim, H, W).to(device))
            backward_initial_states.append(backward_initial_state)
            h_out, forward_c_out, backward_c_out = model.forward(T, test_data, forward_initial_states, backward_initial_states, True)

            total += torch.prod(torch.tensor(labels.shape))
            correct += (h_out == labels).sum().item()

    return 100 * correct / total


def train(trainloader, model, criterion, optimizer, device, devloader):
    for epoch in range(EPOCH):
        running_loss = 0.0
        for i, (data, labels) in enumerate(trainloader):
            data = data.to(device)
            labels = labels.to(device)

            forward_initial_states = []
            forward_initial_state = (torch.rand(batch_size, hidden_dim, H, W).to(device), torch.zeros(batch_size, hidden_dim, H, W).to(device))
            forward_initial_states.append(forward_initial_state)
            backward_initial_states = []
            backward_initial_state = (torch.rand(batch_size, hidden_dim, H, W).to(device), torch.zeros(batch_size, hidden_dim, H, W).to(device))
            backward_initial_states.append(backward_initial_state)

            optimizer.zero_grad()
            h_out, forward_c_out, backward_c_out = model.forward(T, data, forward_initial_states, backward_initial_states, True)
            loss = criterion(h_out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % PRINT_EVERY == (PRINT_EVERY - 1):
                dev_acc = test(devloader, model, device)
                print('[epoch %d, iter %5d] loss: %.3f, dev acc: %.3f %%' %
                      (epoch + 1, i + 1, running_loss / PRINT_EVERY, dev_acc))
                running_loss = 0.0
    print('Finished Training')


def predict(img, num_of_pred, device, model):
    cur_img = img
    pred_img = None

    for i in range(num_of_pred):
        cur_img = cur_img.to(device)

        forward_initial_states = []
        forward_initial_state = (
        torch.rand(batch_size, hidden_dim, H, W).to(device), torch.zeros(batch_size, hidden_dim, H, W).to(device))
        forward_initial_states.append(forward_initial_state)
        backward_initial_states = []
        backward_initial_state = (
        torch.rand(batch_size, hidden_dim, H, W).to(device), torch.zeros(batch_size, hidden_dim, H, W).to(device))
        backward_initial_states.append(backward_initial_state)

        h_out, forward_c_out, backward_c_out = model.forward(T, cur_img, forward_initial_states, backward_initial_states, True)

        pred_img = h_out

        plt.imshow(pred_img[0, 0, 0, :, :].data.cpu().numpy())
        plt.show()

        cur_img = pred_img


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # input_tensor = torch.rand(batch_size,T,input_dim,H,W)

    depth, flow, segm, normal, annotation, img, keypoint = load_data()

    segm = segm.to(dtype=torch.float32)

    segm.unsqueeze_(2)
    segm = segm[:, :120, :, :, :]
    # print(segm.shape)
    segm_predict_train_loader = split_T(segm)
    segm_predict_validation_loader = split_T(segm)
    # segm_predict_test_loader = split_T(segm[:, :120, :, :, :])

    model = BDClstm((H, W), input_dim, hidden_dim)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(segm_predict_train_loader, model, criterion, optimizer, device, segm_predict_validation_loader)

    num_of_pred = segm.shape[1]

    predict(segm[:, :1, :, :, :], num_of_pred, device, model)


if __name__ == '__main__':
    main()