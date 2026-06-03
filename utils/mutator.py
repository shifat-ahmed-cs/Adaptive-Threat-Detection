"""
Threat Mutation Module for Adversarial Learning
Implements strategies to mutate malware API sequences to evade detection.
Ref: Thesis Objective 2 - "Integrate Threat Mutation for Adversarial Learning"
"""

import numpy as np
import torch

class ThreatMutator:
    """
    Generates adversarial examples by mutating API sequences.
    Simulates common evasion techniques like dead code insertion and excessive API noise.
    """
    
    def __init__(self, vocab_size, pad_idx=0):
        """
        Args:
            vocab_size (int): Size of the API vocabulary.
            pad_idx (int): Index used for padding (usually 0).
        """
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

    def mutate_batch(self, sequences, mutation_rate=0.1, strategy='random_insert'):
        """
        Apply mutations to a batch of sequences.
        
        Args:
            sequences (torch.Tensor): Input sequences of shape (batch_size, seq_len)
            mutation_rate (float): Probability of mutating a specific position or fraction of noise to add.
            strategy (str): Mutation strategy ('random_insert', 'noise_append', 'random_replace')
            
        Returns:
            torch.Tensor: Mutated sequences
        """
        if strategy == 'random_insert':
            return self._random_insert(sequences, mutation_rate)
        elif strategy == 'noise_append':
            return self._noise_append(sequences, mutation_rate)
        elif strategy == 'random_replace':
            return self._random_replace(sequences, mutation_rate)
        else:
            raise ValueError(f"Unknown mutation strategy: {strategy}")

    def _random_insert(self, sequences, rate):
        """
        Randomly inserts noise API calls into the sequence.
        Preserves original order but shifts elements.
        """
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        # We'll create a new tensor to hold mutated sequences
        # Note: Implementing true insertion in fixed-length tensors involves shifting/truncating
        mutated_seqs = sequences.clone()
        
        for i in range(batch_size):
            # Convert to numpy for easier manipulation
            seq = sequences[i].cpu().numpy()
            
            # Identify non-padding elements
            non_pad_indices = np.where(seq != self.pad_idx)[0]
            if len(non_pad_indices) == 0:
                continue
                
            actual_len = len(non_pad_indices)
            num_mutations = int(actual_len * rate)
            
            if num_mutations == 0:
                continue
                
            # Create a list to insert into
            seq_list = seq[:actual_len].tolist()
            
            for _ in range(num_mutations):
                # Random position to insert
                insert_pos = np.random.randint(0, len(seq_list) + 1)
                # Random API call (exclude padding)
                noise_api = np.random.randint(1, self.vocab_size)
                seq_list.insert(insert_pos, noise_api)
            
            # Truncate back to original length or pad
            if len(seq_list) > seq_len:
                new_seq = np.array(seq_list[:seq_len])
            else:
                new_seq = np.pad(seq_list, (0, seq_len - len(seq_list)), 'constant', constant_values=self.pad_idx)
                
            mutated_seqs[i] = torch.from_numpy(new_seq).to(device)
            
        return mutated_seqs

    def _average_random_replace(self, sequences, rate):
        """
        Use random_replace instead.
        """
        pass

    def _random_replace(self, sequences, rate):
        """
        Randomly replaces existing API calls with noise.
        This simulates 'polymorphism' where instructions are substituted.
        """
        mask = torch.rand_like(sequences, dtype=torch.float) < rate
        # Don't replace padding
        mask = mask & (sequences != self.pad_idx)
        
        noise = torch.randint(1, self.vocab_size, sequences.shape, device=sequences.device)
        return torch.where(mask, noise, sequences)

    def _noise_append(self, sequences, rate):
        """
        Appends random garbage API calls to the end of the sequence (before padding logic).
        For fixed length, this simply fills the padding area with noise.
        """
        batch_size, seq_len = sequences.shape
        mutated_seqs = sequences.clone()
        
        noise = torch.randint(1, self.vocab_size, sequences.shape, device=sequences.device)
        
        # Replace padding with noise with probability 'rate'
        # This simulates appending junk data at the end of a file/execution
        padding_mask = (sequences == self.pad_idx)
        apply_mask = padding_mask & (torch.rand_like(sequences, dtype=torch.float) < rate)
        
        return torch.where(apply_mask, noise, mutated_seqs)
