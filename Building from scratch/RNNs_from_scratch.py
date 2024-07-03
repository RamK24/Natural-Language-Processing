import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, embedding_dimension, hidden_size, num_cells=1, bidirectional=False):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.hidden_size = hidden_size
        self.num_cells = num_cells
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.tanh = nn.Tanh()

    def init_linear_layers(self):
        input2hidden_fc = nn.Linear(self.embedding_dimension + self.hidden_size, self.hidden_size)
        return [input2hidden_fc] + [nn.Linear(2 * self.hidden_size, self.hidden_size) for i in
                                    range(self.num_cells - 1)]

    def forward(self, x, h0, c0=None):
        self.linear_layers_forward = self.init_linear_layers()
        h0 = torch.permute(h0, (1, 0, 2))
        batch_size, num_hidden_states, _ = x.shape
        time_step_seq = range(0, num_hidden_states, 1)
        hidden_states_forward, ht_forward = self.compute_one_direction(x, h0, time_step_seq, self.linear_layers_forward)
        if self.bidirectional:
            h0_forward, h0_backward = h0[:h0.shape[0] // 2], h0[h0.shape[0] // 2:]
            self.linear_layers_backward = self.init_linear_layers()
            hidden_states_backward, ht_backward = self.compute_one_direction(x, h0,
                                                                             range(num_hidden_states - 1, -1, -1),
                                                                             self.linear_layers_backward)
            return torch.cat((hidden_states_forward, hidden_states_backward), dim=-1), torch.cat(
                (ht_forward, ht_backward), dim=-1)
        return hidden_states_forward, ht_forward

    def compute_one_direction(self, x, h0, time_step_seq, linear_layers):
        batch_size, num_hidden_states, _ = x.shape
        hidden_states = torch.zeros(batch_size, num_hidden_states, self.hidden_size)
        for t in time_step_seq:
            x_t = x[:, t:t + 1, :].clone()
            for layer in range(self.num_cells):
                xt_ht_prev = torch.cat((x_t, h0[:, layer:layer + 1, :].clone()), dim=-1)
                ht = self.tanh(linear_layers[layer](xt_ht_prev))
                x_t = ht
                h0[:, layer:layer + 1, :] = ht.clone()
            hidden_states[:, t:t + 1, :] = ht.clone()
        return hidden_states, ht


class GRU(nn.Module):
    def __init__(self, embedding_dimension, hidden_size, num_cells=1, bidirectional=False):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.hidden_size = hidden_size
        self.num_cells = num_cells
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.update_gate = [nn.Linear(embedding_dimension + self.hidden_size, self.hidden_size)] + [
            nn.Linear(2 * self.hidden_size, self.hidden_size) for _ in range(num_cells - 1)]
        self.reset_gate = [nn.Linear(embedding_dimension + self.hidden_size, self.hidden_size)] + [
            nn.Linear(2 * self.hidden_size, self.hidden_size) for _ in range(num_cells - 1)]
        self.new_gate = [nn.Linear(embedding_dimension + self.hidden_size, self.hidden_size)] + [
            nn.Linear(2 * self.hidden_size, self.hidden_size) for _ in range(num_cells - 1)]

    def forward(self, x, h0, c0=None):
        time_step_seq = range(0, x.shape[1], 1)
        hidden_states_forward, ht_forward = self.compute_one_direction(x, h0, time_step_seq)
        if self.bidirectional:
            h0_backward = h0[h0.shape[0] // 2:]
            time_step_seq = range(self.num_cells - 1, -1, -1)
            hidden_states_backward, ht_backward = self.compute_one_direction(x, h0_backward, time_step_seq)
            return torch.cat((hidden_states_forward, hidden_states_backward), dim=-1), torch.cat(
                (ht_forward, ht_backward), dim=-1)
        return hidden_states_forward, ht_forward

    def compute_one_direction(self, x, h0, time_step_seq):
        h0 = torch.permute(h0, (1, 0, 2))
        batch_size, num_time_steps, _ = x.shape
        hidden_states = torch.zeros(batch_size, num_time_steps, self.hidden_size)
        for t in time_step_seq:
            x_t = x[:, t:t + 1, :]
            for layer in range(self.num_cells):
                h_prev = h0[:, layer:layer + 1, :].clone()
                xt_ht = torch.cat((x_t, h_prev), dim=-1)
                rt = self.sigmoid(self.update_gate[layer](xt_ht))
                zt = self.sigmoid(self.reset_gate[layer](xt_ht))
                xt_ht_rt = torch.cat((x_t, rt * h_prev), dim=-1)
                nt = self.tanh(self.new_gate[layer](xt_ht_rt))
                ht = (1 - zt) * nt + zt * h_prev
                x_t = ht
                h0[:, layer:layer + 1, :] = ht
            hidden_states[:, t:t + 1, :] = ht
        return hidden_states, ht


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_cells, bidirectional=False):
        super().__init__()
        self.embedding_dimension = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_cells = num_cells
        self.bidirectional = bidirectional
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.update_gate = [nn.Linear(self.embedding_dimension + hidden_dim, hidden_dim)] + [
            nn.Linear(2 * self.hidden_dim, self.hidden_dim) for i in range(self.num_cells - 1)]
        self.forget_gate = [nn.Linear(self.embedding_dimension + hidden_dim, hidden_dim)] + [
            nn.Linear(2 * self.hidden_dim, self.hidden_dim) for i in range(self.num_cells - 1)]
        self.output_gate = [nn.Linear(self.embedding_dimension + hidden_dim, hidden_dim)] + [
            nn.Linear(2 * self.hidden_dim, self.hidden_dim) for i in range(self.num_cells - 1)]
        self.memory_cell_candidate = [nn.Linear(self.embedding_dimension + hidden_dim, hidden_dim)] + [
            nn.Linear(2 * self.hidden_dim, self.hidden_dim) for i in range(self.num_cells - 1)]

    def forward(self, x, h0, c0):
        time_step_seq = range(0, self.num_cells, 1)
        hidden_states_forward, ht_forward = self.compute_one_direction(x, h0, c0, time_step_seq)
        if self.bidirectional:
            h0_backward = h0[h0.shape[0] // 2:]
            time_step_seq = range(self.num_cells - 1, -1, -1)
            hidden_states_backward, ht_backward = self.compute_one_direction(x, h0_backward, c0, time_step_seq)
            return (torch.cat((hidden_states_forward[0], hidden_states_backward[0]), dim=-1),
                    torch.cat((hidden_states_forward[1], hidden_states_backward[1]), dim=-1)), (torch.cat(
                (ht_forward[0], ht_backward[1]), dim=-1), torch.cat((ht_forward[0], ht_backward[1]), dim=-1))
        return hidden_states_forward, ht_forward

    def compute_one_direction(self, x, h0, c0, time_step_seq):
        h0 = torch.permute(h0, (1, 0, 2))
        c0 = torch.permute(c0, (1, 0, 2))
        batch_size, num_time_steps, _ = x.shape
        hidden_states = torch.zeros(batch_size, num_time_steps, self.hidden_dim)
        memory_states = torch.zeros(batch_size, num_time_steps, self.hidden_dim)
        for t in time_step_seq:
            xt = x[:, t:t + 1, :]
            for layer in range(self.num_cells):
                h_prev = h0[:, layer:layer + 1, :].clone()
                c_prev = c0[:, layer:layer + 1, :].clone()
                xt_h_prev = torch.cat((xt, h_prev), dim=-1)
                ut = self.sigmoid(self.update_gate[layer](xt_h_prev))
                ft = self.sigmoid(self.forget_gate[layer](xt_h_prev))
                ot = self.sigmoid(self.output_gate[layer](xt_h_prev))
                c_tilde_t = self.tanh(self.memory_cell_candidate[layer](xt_h_prev))
                c_t = ft * c_prev + ut * c_tilde_t
                h_t = ot * self.tanh(c_t)
                h0[:, layer:layer + 1, :] = h_t
                c0[:, layer:layer + 1, :] = c_t
                xt = h_t
            hidden_states[:, t:t + 1, :] = h_t
            memory_states[:, t:t + 1, :] = c_t
        return (hidden_states, memory_states), (h_t, c_t)