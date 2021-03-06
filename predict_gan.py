
# coding: utf-8

# In[2]:


import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.optim as optim


# In[5]:


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


# In[10]:


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
                              
    def forward(self, T, input_tensor, forward_initial_states,backward_initial_states):
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
                              
        forward_h_out,forward_c_out = self.forward_lstm(input_tensor, forward_initial_states)
        forward_h_out = forward_h_out[0]
        forward_c_out = forward_c_out[0]
        #invert input_tensor along time step
        idx = [i for i in range(input_tensor.size(1)-1,-1,-1)]
        # print(len(idx))
        idx = torch.cuda.LongTensor(idx)
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
        '''

        pred_img = torch.abs(h_out)  # .data > 0.5) * 1.0  # .float()

        # .to(dtype=torch.float64)
        avg = torch.mean(pred_img) * 1.01

        zeros = torch.zeros(pred_img.shape)
        # ones = torch.ones(plot_img.shape)
        out_img = torch.where(pred_img > avg, pred_img, zeros)
        '''
        
        return h_out,forward_h_out,inverted_h_out
        


# In[37]:


class Generator2(nn.Module):

    def __init__(self, kernel_size=3):
        super(Generator2, self).__init__()
        #input size (6,256,256)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6,32,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32,32,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(3,stride=2,padding=1))
        #(3,128,128)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(3,stride=2,padding=1))
        #(64,64,64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128,128,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(3,stride=2,padding=1))
        #(128,32,32)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256,256,kernel_size,stride = 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(3,stride=2,padding=1))
        #(256,16,16)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(128,128,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        #(128,32,32)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256,64,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        #(64,64,64)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128,32,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32,32,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        #(3,256,256)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64,3,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(3,3,kernel_size,stride = 1, padding=2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        
    def forward(self,inputs,base_img):
        N,T,_,_,_ = inputs.size()
        base_img = base_img.view(N,1,3,64,64).repeat(1,T,1,1,1)
        cat_inputs = torch.cat((inputs,base_img),dim=2)
        #print(cat_inputs.size())
        cat_inputs = cat_inputs.view(N*T,6,64,64)
        e1 = self.conv1(cat_inputs)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        d4 = self.deconv4(e4)
        d3_in = torch.cat((e3,d4),1)
        d3 = self.deconv3(d3_in)
        d2_in = torch.cat((e2,d3),1)
        d2 = self.deconv2(d2_in)
        d1_in = torch.cat((e1,d2),1)
        d1 = self.deconv1(d1_in)
        out = d1.view(N,T,3,64,64)
        return out


# In[38]:


#test Generator
inputs = torch.rand((10,20,3,64,64))
base_img = torch.rand((10,3,64,64))
model = Generator2()
outputs = model.forward(inputs,base_img)
#print(outputs.size())


# In[19]:


class Discriminator(nn.Module):

    def __init__(self, kernel_size=5, dim=64):
        super(Discriminator, self).__init__()
        self.kernel_size = kernel_size
        self.dim = dim
        self.conv1 = nn.Conv2d(3, self.dim, self.kernel_size, stride=2, padding=2)
        #(64,32,32)
        #LeakyRelu
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim,kernel_size,stride=2,padding=2)
        self.bn1 = nn.BatchNorm2d(2*self.dim)
        #LeakyRelu
        #(128,16,16)
        self.conv3 = nn.Conv2d(2*self.dim,4*self.dim,kernel_size,stride=2,padding=2)
        self.bn2 = nn.BatchNorm2d(4*self.dim)
        #LeakyRelu
        #(256,8,8)
        self.conv4 = nn.Conv2d(4*self.dim,8*self.dim,kernel_size,stride=2,padding=2)
        self.bn3 = nn.BatchNorm2d(8*self.dim)
        #LeakyRelu
        #(512,4,4)
        self.conv5 = nn.Conv2d(8*self.dim,8*self.dim,kernel_size,stride=2,padding=2)
        self.bn4 = nn.BatchNorm2d(8*self.dim)
        #LeakyRelu
        #(512,8,10)
        self.fc = nn.Linear(4*4*512,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, channel, height, width = x.shape
        output = self.conv1(x)
        output = F.leaky_relu(output)
        output = self.conv2(output)
        output = self.bn1(output)
        output = F.leaky_relu(output)
        output = self.conv3(output)
        output = self.bn2(output)
        output = F.leaky_relu(output)
        output = self.conv4(output)
        output = self.bn3(output)
        output = F.leaky_relu(output)
       
        
        output = output.view(-1, 4*4*512)
        
        output = self.fc(output)
        output = self.sigmoid(output)
        
        return output
      

class Appearance_D(nn.Module):

    def __init__(self,kernel_size=4):
        super(Appearance_D, self).__init__()
        #network 1
        self.network1_l1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(64,affine=False),
            nn.LeakyReLU())
        self.network1_l2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(128,affine=False),
            nn.LeakyReLU())
        self.network1_l3 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.LeakyReLU())
        
        #network 2
        #input to 2: (256*4,8,8)
        
        self.network2 = nn.Sequential(
            nn.ConvTranspose2d(256*4,256,3,padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.LeakyReLU(),      
            nn.Conv2d(256,512,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(512,affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(512,1024,kernel_size,stride=4),
            nn.BatchNorm2d(1024,affine=False),
            nn.LeakyReLU())
        self.fc = nn.Sequential(
            nn.Linear(1024,64),
            nn.LeakyReLU(),
            nn.Linear(64,1),
            nn.Sigmoid())
      
    def forward(self,x,y_a):
        '''
        -input x(N,T,3,64,64)
               y(N,1,3,64,64)
        -output predictions torch tensor of (N)
        '''
        N,T,_,_,_ = x.size()

        y_out = self.network1_l1(y_a) #(N,256,8,8)
        y_out = self.network1_l2(y_out) #(N,256,8,8)
        y_out = self.network1_l3(y_out) #(N,256,8,8)
        
        #print(y_out.size())
        predictions = torch.zeros(N,T-3).cuda()
        layer1_out = torch.zeros(N,T-3,64,32,32).cuda()
        layer2_out = torch.zeros(N,T-3,128,16,16).cuda()
        for t in range(T-3):
            #x_temp (N,3,64,64)
            x_temp_1 = x[:,t,:,:,:]
            x_temp_2 = x[:,t+1,:,:,:]
            x_temp_3 = x[:,t+2,:,:,:]
            x_in = torch.cat((x_temp_1,x_temp_2,x_temp_3),0)
            #x_in (3*N,3,64,64)

            x_out_1 = self.network1_l1(x_in) #(3*N,256,8,8)
            layer1_out[:,t,:,:,:] = x_out_1[:N]
            x_out_2 = self.network1_l2(x_out_1)
            layer2_out[:,t,:,:,:] = x_out_2[:N]
            
            x_out = self.network1_l3(x_out_2)
            #print(x_out.size())
            x_out_1 = x_out[:N]
            x_out_2 = x_out[N:2*N]
            x_out_3 = x_out[2*N:3*N]
            
            #print("sizes", x_out_1.size(), x_out_2.size(), x_out_3.size(), y_out.size())

            x_1 = torch.cat((x_out_1,x_out_2,x_out_3,y_out),1) #(N,256*4,8,8)
            x_2 = self.network2(x_1)
            
            x_2 = torch.squeeze(x_2) #(N,1024)
            out = self.fc(x_2) #(N,1)
            predictions[:,t] = torch.squeeze(out)
            
        predictions = torch.mean(predictions,1)
        #prediction is the average across all time steps
            
        return predictions,layer1_out,layer2_out
        
# In[11]:


class Motion_D(nn.Module):

    def __init__(self,q,num_classes,kernel_size=4):
        super(Motion_D, self).__init__()
        #input y_a (3*64*64) 
        #y_l (c*4*4) ??
        #x (3*64*64)
        self.q = q
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size,stride=2,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(128,affine=False),
            nn.LeakyReLU())
        self.encoder2 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.LeakyReLU())
        
        #ConvLSTM(input_size,input_dim,hidden_dim,kernel_size,num_layer)
        self.lstm = ConvLSTM((8,8), 256, 256, (3,3), 1)
        
        #last hidden state h_out (N*256*8*8)
        self.conv4 = nn.Conv2d(256,64,kernel_size,stride=2,padding=1)
        self.bn4 = nn.BatchNorm2d(64,affine=False)
        #LeakyRelu -> flatten
        self.fc_h1 = nn.Linear(1024,64)
        self.bn_fc1 = nn.BatchNorm1d(64,affine=False)
        #LeakyRelu
        self.fc_h2 = nn.Linear(64,num_classes)
        self.bn_fc2 = nn.BatchNorm1d(num_classes,affine=False)
        #LeakyRelu -> softmax
        
        #output: ot (N,256,8,8)
        self.conv6 = nn.Conv2d(256,64,kernel_size,stride=2,padding=1)
        self.bn6 = nn.BatchNorm2d(64,affine=False)
        #leakyRelu -> flatten
        self.fc_o = nn.Linear(1024,q)
        
        #concatenate y_l (c*4*4) + (64*4*4) = (c+64,4,4)
        self.conv5 = nn.Conv2d(num_classes+64,64,kernel_size,stride=2)
        self.bn5 = nn.BatchNorm1d(64,affine=False)
        #leakyrelu
        self.fc_y = nn.Linear(64,1)
        #sigmoid
        
        #TODO forward
    def forward(self,x,y_a,y_l):
        N = x.size()[0]
        T = x.size()[1]
        # x (N,T,3,64,64)
        # y_a (N,3,64,64)
        #y_l = (N,C)
        x = x.view(-1,3,64,64)
        
        x_encoded_1 = self.encoder1(x)
        layer1_out = x_encoded_1.view(N,T,128,16,16)
        x_encoded = self.encoder2(x_encoded_1)

        y_encoded_1 = self.encoder1(y_a)
        y_encoded = self.encoder2(y_encoded_1)
        
        x_encoded = x_encoded.view(N,T,256,8,8)
        
        initial_states = []
        #TODO: check if initial cell state is 0?
        initial_cell = Variable(torch.zeros_like(y_encoded)).cuda()
        initial_states.append((y_encoded,initial_cell))
        
        
        o,h_last = self.lstm(x_encoded, initial_states)
        o = o[0]
        h_last = h_last[0]
        
        h_out = self.conv4(h_last)
        h_out = self.bn4(h_out)
        h_out = F.leaky_relu(h_out)
        
        #print('h_out shape',h_out.size())
        l_out = h_out.view(N,-1)
        l_out = self.fc_h1(l_out)
        l_out = self.bn_fc1(l_out)
        l_out = F.leaky_relu(l_out)
        l_out = self.fc_h2(l_out)
        l_out = self.bn_fc2(l_out)
        l_out = F.softmax(F.leaky_relu(l_out))
        #y_m_l output (label predicted)
        
        #concatenate y_l and h_out
        #print('yl',y_l.shape)
        y_l = y_l.view(N,-1,1,1).repeat(1,1,4,4)
        #print('y_l shape ', y_l.size())
        predict = torch.cat((y_l,h_out),1)
        #print('predict shape', predict.size())
        predict = self.conv5(predict)
        predict = torch.squeeze(predict)
        #print('predict shape after squeeze',predict.size())
        predict = self.bn5(predict)
        predict = F.leaky_relu(predict)
        predict = self.fc_y(predict)
        predict_out = F.sigmoid(predict)
        
        return predict_out,l_out,layer1_out,x_encoded
        

class Decoder(nn.Module):

    def __init__(self,p,kernel_size=3,upsample_size=2):
        super(Decoder, self).__init__()
        #convLSTM out: (N,256,8,8)
        #encoder out: (N,256,8,8)
        #noise z : (N,p,8,8)
        #input size:(N,512+p,8,8)

        self.s1 = Gate(2,3,256)
        self.s2 = Gate(4,3,128)
        self.s3 = Gate(8,3,64)

        self.conv6 = nn.Conv2d(256*2+p,256,kernel_size,padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        #LeakyReLU
        self.upsample = nn.Upsample(scale_factor=upsample_size, mode='bilinear')
        #Gating  is it gonna backprop?
        
        self.conv7 = nn.Conv2d(256,128,kernel_size,padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        #LeakyReLU
        #upsample
        #Gating
        
        self.conv8 = nn.Conv2d(128,64,kernel_size,padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        #LeakyReLU
        #upsample
        #Gating
        
        self.conv9 = nn.Conv2d(64,64,kernel_size,padding=1)
        self.bn9 = nn.BatchNorm2d(64)
        #LeakyReLU
        self.conv10 = nn.Conv2d(64,3,kernel_size,padding=1)
        #Tanh
        
    def forward(self,lstm_out,z,encoder_out,encoder_cache):
        #z - tiling noise size(p*8*8)
        #lstm_out:(N,512+P,8,8)
        
        e3,e2,e1 = encoder_cache
        decoder_input = torch.cat((lstm_out,encoder_out,z),1)
        u1 = self.conv6(decoder_input)
        u1 = self.bn6(u1)
        u1 = F.leaky_relu(u1)
        u1 = self.upsample(u1)

        s1 = self.s1(lstm_out)
        u2 = s1*u1 + (1-s1)*e1
        
        u2 = self.conv7(u2)
        u2 = self.bn7(u2)
        u2 = F.leaky_relu(u2)
        u2 = self.upsample(u2)

        s2 = self.s2(lstm_out)
        
        u3 = s2*u2 + (1-s2)*e2
        
        u3 = self.conv8(u3)
        u3 = self.bn8(u3)
        u3 = F.leaky_relu(u3)
        u3 = self.upsample(u3)
        
        s3 = self.s3(lstm_out)
        u4 = s3*u3 + (1-s3)*e3
        u4 = self.conv9(u4)
        u4 = self.bn9(u4)
        u4 = F.leaky_relu(u4)
        u4 = self.conv10(u4)
        out = F.tanh(u4)
        return out   


# In[21]:

class Gate(nn.Module):

    def __init__(self, upsample_size,kernel_size,num_filters):
        super(Gate, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample_size, mode='bilinear')
        self.conv = nn.Conv2d(256,num_filters,kernel_size,padding=1)
    def forward(self,lstm_out):
        h = self.upsample(lstm_out)
        
        h = self.conv(h)
        h = F.leaky_relu(h)
        out = F.sigmoid(h)
        return out
        


class Generator1(nn.Module):

    def __init__(self,batch_size,q,p,kernel_size=3):
        super(Generator1, self).__init__()
        self.num_keypoints = q
        self.p = p
        self.encode1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,kernel_size,padding=1),
            nn.BatchNorm2d(64,affine=False),
            nn.LeakyReLU())
        
        self.encode2 = nn.Sequential(
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(64,128,kernel_size,padding=1),
            nn.BatchNorm2d(128,affine=False),
            nn.LeakyReLU())
        
        self.encode3 = nn.Sequential(
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(128,256,kernel_size,padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.LeakyReLU()
        )
        self.encode4 = nn.Sequential(
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(256,256,kernel_size,padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.LeakyReLU()
        )
        
        self.fc_q = nn.Linear(q,q)
        
        #ConvLSTM(input_size,input_dim,hidden_dim,kernel_size,num_layer)
        self.lstm = ConvLSTM((8,8), q, 256, (3,3), 1)
        
        self.decode = Decoder(p,kernel_size,upsample_size=2)
        
        
    def forward(self,first_frame,keypoints,z):
        N,T,_ = keypoints.size()
        
        #middle_tensor = torch.zeros(N,T,self.num_keypoints).cuda()
        #input_tensor = torch.cat((prev_keypoints,middle_tensor,post_keypoints),dim=1)
        #(N,3*T,q)
        
        #tile y_v
        q = self.fc_q(keypoints)
        keypoints = F.sigmoid(q)
        #y_v = t * q + (1-t)*q
        
        #TODO: check implementation of tiling
        keypoints = keypoints.view(N,-1,self.num_keypoints,1,1).repeat(1,1,1,8,8)
        
        #encoder
        e1_first = self.encode1(first_frame)
        e2_first = self.encode2(e1_first)
        e3_first = self.encode3(e2_first)
        e_out_first = self.encode4(e3_first)
        
        
        
        #initial_states = []
        forward_initial_states = []
        #TODO: check if initial cell state is 0?
        #print(e_out.size())
        initial_cell = Variable(torch.zeros_like(e_out_first))
        forward_initial_states.append((e_out_first,initial_cell))
        #print(y_v.size())
        
        lstm_out,_ = self.lstm(keypoints,forward_initial_states)
        lstm_out = lstm_out[0]
        #lstm_out (N,3*T,256,8,8)
        
        
        #create random noise z
        #z = torch.rand((N,p))
        z = z.view(N,self.p,1,1).repeat(1,1,8,8)
        
        
        decode_out = torch.zeros(N,T,3,64,64).cuda()
        #decode each time step output
        for t in range(lstm_out.size()[1]):
            decode_out[:,t,:,:,:] = self.decode(lstm_out[:,t,:,:,:],z,e_out_first,(e1_first,e2_first,e3_first))
        
        return decode_out
        


