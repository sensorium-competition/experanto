"""
Tests for SessionBatchSampler, SessionSpecificSampler, and FastSessionDataLoader functionality.
"""
import pytest
from typing import Dict, Optional, Any
from copy import deepcopy
from collections import defaultdict, Counter
import math
import numpy as np
import torch

from experanto.utils import (
    count_batches,
    SessionConcatDataset, 
    SessionBatchSampler, 
    SessionSpecificSampler, 
    FastSessionDataLoader
)


def states_equal(state1, state2):
    """
    Compare two nested state dictionaries that may contain numpy arrays.
    
    This handles the case where RNG states contain numpy arrays that can't be
    compared directly with == due to ambiguous truth values.
    """
    if type(state1) != type(state2):
        return False
    
    if isinstance(state1, dict):
        if set(state1.keys()) != set(state2.keys()):
            return False
        return all(states_equal(state1[k], state2[k]) for k in state1.keys())
    
    elif isinstance(state1, (list, tuple)):
        if len(state1) != len(state2):
            return False
        return all(states_equal(a, b) for a, b in zip(state1, state2))
    
    elif isinstance(state1, np.ndarray):
        return isinstance(state2, np.ndarray) and np.array_equal(state1, state2)
    
    else:
        # For primitives and other types
        return state1 == state2


def _get_sampler_attributes(
    _fast_dataloader: FastSessionDataLoader,
    _session_name: Optional[str] = None,
    _attr_name: str = 'indices',
):
    _dummy_dl = list(_fast_dataloader.session_dataloaders.values())[0]
    assert hasattr(_dummy_dl.batch_sampler, _attr_name), \
        f"Session dataloader {_attr_name} attribute not found"
    ret = {
        k: deepcopy(getattr(v.batch_sampler, _attr_name))
        for k, v in _fast_dataloader.session_dataloaders.items()
    }
    if _session_name is not None:
        assert _session_name in ret, f"Session name {_session_name} not found in dataloader"
        ret = {k: v for k, v in ret.items() if k == _session_name}
    return ret


def _update_state(
    _prev: Dict[str, Any],
    _curr: Dict[str, Any],
    _validate_change: bool = True,
    _session_name: Optional[str] = None,
) -> Dict[str, Any]:
    if _session_name is not None:
        assert _session_name in _prev and _session_name in _curr, \
            f"Session name {_session_name} not found in state!"
        _curr = {k: v for k, v in _curr.items() if k == _session_name}
    if _validate_change:
        assert not states_equal({k:v for k,v in _prev.items() if k in _curr},  _curr), \
            "State did not change!"
    _prev.update(_curr)
    return _prev


def _check_reset(_fast_dataloader: FastSessionDataLoader):
    """Utility function to check that dataloader state was reset."""
    assert _fast_dataloader.current_batch == 0
    assert _fast_dataloader.position_in_epoch == 0
    assert all(p == 0 for p in _fast_dataloader.session_positions.values())
    assert all(dl.batch_sampler.position == 0 for dl in _fast_dataloader.session_dataloaders.values())
    assert all(p == 0 for p in _fast_dataloader.batches_from_session.values())
    assert len(_fast_dataloader.active_sessions) == len(_fast_dataloader.session_dataloaders)
    assert len(_fast_dataloader.batch_sampler.consumed_sessions) == 0


@pytest.fixture
def test_datasets(
    mock_dataset,
    seed: int = 42,
):
    """Create test datasets with different lengths using the conftest MockDataset."""
    # Create datasets with specific lengths and fixed session names for testing
    datasets = [
        mock_dataset(min_length=5, max_length=5, seed=seed, session_name="session_a"),  # Length 5
        mock_dataset(min_length=8, max_length=8, seed=(seed + 1), session_name="session_b"),  # Length 8
        mock_dataset(min_length=10, max_length=10, seed=(seed + 2), session_name="session_c") # Length 10
    ]
    
    session_names = ["session_a", "session_b", "session_c"]
    return datasets, session_names


@pytest.fixture
def concat_dataset(test_datasets):
    """Create a SessionConcatDataset from test datasets."""
    datasets, session_names = test_datasets
    return SessionConcatDataset(datasets, session_names)


@pytest.fixture
def batch_sampler_factory(concat_dataset):
    def _factory(
        batch_size: int = 2,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 12345,
    ):
        return SessionBatchSampler(
            dataset=concat_dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed
        )
    return _factory


@pytest.fixture
def session_specific_sampler_factory(
    batch_sampler_factory,
):
    """A factory to create session-specific samplers."""
    def _factory(
        shuffle: bool = True,
        session_name: str = 'session_a',
        seed: int = 12345,
    ):
        batch_sampler = batch_sampler_factory(
            shuffle=shuffle,
            seed=seed
        )
        assert session_name in batch_sampler.session_names, (
            f"Session name {session_name} not found in batch sampler"
        )
        return SessionSpecificSampler(
            indices=batch_sampler.session_indices[session_name],
            batch_size=batch_sampler.batch_size,
            drop_last=batch_sampler.drop_last,
            shuffle=batch_sampler.shuffle,
            seed=seed
        )
    return _factory


@pytest.fixture
def fast_dataloader_factory(concat_dataset):
    """A factory to create FastSessionDataLoader"""
    def _factory(
        batch_size: int = 2,
        drop_last: bool = False,
        shuffle: bool = True,
        cycle_mode: str = 'balanced',
        seed: int = 12345,
    ):
        return FastSessionDataLoader(
            dataset=concat_dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            cycle_mode=cycle_mode,
            seed=seed
        )
    return _factory


class TestSessionConcatDataset:
    """Test SessionConcatDataset functionality."""
    
    def test_basic_functionality(self, test_datasets, concat_dataset):
        """Test basic SessionConcatDataset operations."""
        _, session_names = test_datasets

        # Check total length
        assert len(concat_dataset) == 5 + 8 + 10  # 23
        
        # Check session names
        assert concat_dataset.session_names == session_names
        
        # Check session indices mapping
        expected_indices = {
            "session_a": (0, 5),
            "session_b": (5, 13), 
            "session_c": (13, 23)
        }
        assert concat_dataset.session_indices == expected_indices
    
    def test_get_session_for_idx(self, concat_dataset):
        """Test session retrieval by index."""
        # Test various indices
        assert concat_dataset.get_session_for_idx(0) == "session_a"
        assert concat_dataset.get_session_for_idx(4) == "session_a"
        assert concat_dataset.get_session_for_idx(5) == "session_b"
        assert concat_dataset.get_session_for_idx(12) == "session_b"
        assert concat_dataset.get_session_for_idx(13) == "session_c"
        assert concat_dataset.get_session_for_idx(22) == "session_c"


class TestSessionBatchSampler:
    """Test SessionBatchSampler functionality."""
    
    def test_initialization(self, concat_dataset, batch_sampler_factory):
        """Test SessionBatchSampler initialization."""
        # Create batch sampler
        batch_sampler = batch_sampler_factory()

        assert batch_sampler.session_names == ["session_a", "session_b", "session_c"]

        session_indices = list(map(lambda x: list(range(*x)), concat_dataset.session_indices.values()))
        session_a_batches = count_batches(session_indices[0], 2, False)
        session_b_batches = count_batches(session_indices[1], 2, False)
        session_c_batches = count_batches(session_indices[2], 2, False)
        assert len(batch_sampler) == (
            session_a_batches + session_b_batches + session_c_batches
        )
        
        # Check batches per session
        expected_batches = {
            "session_a": session_a_batches,
            "session_b": session_b_batches,
            "session_c": session_c_batches
        }
        assert batch_sampler.batches_per_session == expected_batches
    
    def test_session_cycle_generation(self, batch_sampler_factory):
        """Test session cycle order generation."""
        # Create batch sampler
        batch_sampler = batch_sampler_factory()

        # Get multiple cycles - should be shuffled but contain all sessions
        cycle1 = batch_sampler.get_session_cycle()
        cycle2 = batch_sampler.get_session_cycle()
        
        # Each cycle should contain all sessions
        assert set(cycle1) == {"session_a", "session_b", "session_c"}
        assert set(cycle2) == {"session_a", "session_b", "session_c"}
        
        # With different random states, order should be different (though with small
        #  sample, might occasionally be same, so we check 100 times). Also check that
        #  RNG state changes on each call to `get_session_cycle`.
        shuffle_flag = False
        prev_cycle = cycle2
        for _ in range(100):
            prev_prv_rng_state = deepcopy(batch_sampler.prv_rng_state)
            curr_cycle = batch_sampler.get_session_cycle()
            next_prv_rng_state = deepcopy(batch_sampler.prv_rng_state)
            assert not states_equal(prev_prv_rng_state, next_prv_rng_state), \
                "RNG state did not change on call to `get_session_cycle`"
            if not all(p == c for c, p in zip(curr_cycle, prev_cycle)):
                shuffle_flag = True
                break  # Order has changed, shuffling is working.
            prev_cycle = curr_cycle
        # This block executes if the loop completes without a `break`.
        assert shuffle_flag, (
            "Session cycle order did not change over 100 consecutive "
            "generations, suggesting that shuffling is not working."
        )


class TestSessionSpecificSampler:
    """Test SessionSpecificSampler functionality."""
    
    def test_basic_functionality(self, session_specific_sampler_factory):
        """Test basic SessionSpecificSampler operations."""
        # Create a sampler using the factory
        sampler = session_specific_sampler_factory(shuffle=False)

        assert (
            len(sampler) == 
            sampler.num_batches == 
            count_batches(
                sampler.indices,
                sampler.batch_size,
                sampler.drop_last
            )
        )
    
    def test_iteration_without_shuffle(self, session_specific_sampler_factory):
        """Test iteration without shuffling."""
        # Create a sampler using the factory with batch_size=1 for predictable testing
        sampler = session_specific_sampler_factory(shuffle=False)

        # Collect all batches of indices
        batches = list(sampler)
        
        # The sampler should return indices from session_a (which are [0,1,2,3,4] in the concat dataset)
        # With batch_size=2, we expect: [[0,1], [2,3], [4]]
        expected_batches = [[0, 1], [2, 3], [4]]
        assert batches == expected_batches
    
    def test_state_management(self, session_specific_sampler_factory):
        """Test state save/load functionality."""
        sampler = session_specific_sampler_factory()
        
        # Get initial state
        initial_state = sampler.get_state()
        
        # Consume some batches
        iterator = iter(sampler)
        batch1 = next(iterator)
        batch2 = next(iterator)
        
        # Save state after consuming 2 batches
        mid_state = sampler.get_state()
        assert mid_state['position'] == 2
        
        # Continue and collect remaining batches
        remaining_batches_original = list(iterator)
        
        # Create new sampler and restore state
        new_sampler = session_specific_sampler_factory(
            seed=11111  # Different seed to ensure state restoration works
        )
        new_sampler.set_state(mid_state)
        
        # Get remaining batches from restored sampler
        remaining_batches_restored = list(new_sampler)
        
        # Should match exactly
        assert remaining_batches_original == remaining_batches_restored


class TestBalancedFastSessionDataLoader:
    """Test FastSessionDataLoader functionality with cycle_mode='balanced'."""
    
    def test_initialization(self, fast_dataloader_factory):
        """Test FastSessionDataLoader initialization."""
        fast_dataloader = fast_dataloader_factory(batch_size=1)
        assert len(fast_dataloader.session_names) == 3
        assert fast_dataloader.cycle_mode == 'balanced'
        assert fast_dataloader.max_batches_per_session == 10
        # Total batches should be 3 sessions * 10 max_batches = 30
        assert len(fast_dataloader) == 30
    
    def test_main_behavior_cycle_max(self, fast_dataloader_factory):
        """Test main expected behavior with cycle_mode='balanced'."""
        fast_dataloader = fast_dataloader_factory(batch_size=1)
        # Collect all batches from one epoch
        epoch_batches = []
        session_batch_count = defaultdict(int)
        
        for session_name, batch_data in fast_dataloader:
            epoch_batches.append((session_name, batch_data))
            session_batch_count[session_name] += 1
        
        # Should have exactly 30 batches total (3 * 10)
        assert len(epoch_batches) == 30
        
        # Each session should appear exactly 10 times
        assert session_batch_count["session_a"] == 10
        assert session_batch_count["session_b"] == 10  
        assert session_batch_count["session_c"] == 10
        
        # Verify that every 3 consecutive batches contain exactly one from each session
        for i in range(0, 30, 3):
            # Get sessions for batches i, i+1, i+2
            cycle_sessions = {epoch_batches[i][0], epoch_batches[i+1][0], epoch_batches[i+2][0]}
            assert cycle_sessions == {"session_a", "session_b", "session_c"}
    
    def test_shorter_datasets_repeat(self, fast_dataloader_factory):
        """Test that shorter datasets repeat their data when exhausted."""
        fast_dataloader = fast_dataloader_factory(batch_size=1)
        # Track data from session_a (length=5) across the full epoch
        session_batches = defaultdict(list)
        session_responses_hashes = defaultdict(list)
        
        for session_name, batch_data in fast_dataloader:
            # Use hash of responses tensor to identify unique batches
            responses_hash = hash(batch_data['responses'].flatten().tolist().__str__())
            session_batches[session_name].append(batch_data)
            session_responses_hashes[session_name].append(responses_hash)
        
        # Should have 5 batches from the all sessions
        assert all(len(sb) == 10 for sb in session_batches.values())
        
        # Since shortest session only has 5 unique batches, some should repeat
        for session_name, session_hashes in session_responses_hashes.items():
            expected_batches = len(fast_dataloader.session_dataloaders[session_name])
            assert len(set(session_hashes)) == expected_batches
            hash_counts = Counter(session_hashes)
            assert sum(hash_counts.values()) == 10
            # shortest session should have exactly 2 batches per hash
            if session_name == "session_a":
                assert all(count == 2 for count in hash_counts.values())
            # middle session should have 1-2 batches per hash
            elif session_name == "session_b":
                assert all(1 <= count <= 2 for count in hash_counts.values())
            # longest session should have exactly 1 batch per hash
            elif session_name == "session_c":
                assert all(count == 1 for count in hash_counts.values())
    
    def test_synced_epoch_reset_behavior(self, fast_dataloader_factory):
        """Test that dataloader resets after epoch completion.
        
        TODO: Consider adding a few more validation checks
        """    
        # Create dataloader and get synced length
        fast_dataloader = fast_dataloader_factory()
        # Create extended length (imitating 
        #  `unraveling.distributed.utils.SyncedFastSessionDataLoader`)
        synced_len = int(len(fast_dataloader) * 1.5)
        # Get sessions list
        sessions = fast_dataloader.session_names
        # Run for 2 (extended) epochs
        epochs = 2
        while epochs > 0:
            # Initialize epoch counters
            epoch_step = 0
            epoch_batches_per_session = defaultdict(int)
            epoch_batch_hashes_per_session = defaultdict(list)
            # Initialize expected tracker values for this epoch
            expected_current_batch = 0
            expected_position_in_epoch = 0
            expected_session_batch_counts = defaultdict(int)
            expected_session_positions = {k: 0 for k in sessions}
            expected_consumed_sessions = []
            expected_sampler_indices = _get_sampler_attributes(
                fast_dataloader, _attr_name='indices'
            )
            expected_prv_rng_states = _get_sampler_attributes(
                fast_dataloader, _attr_name='prv_rng_state'
            )
            # Run epoch for `synced_len` batches
            dl_iter = iter(fast_dataloader)
            while epoch_step < synced_len:
                try:
                    # Get next batch
                    key, batch = next(dl_iter)
                except StopIteration:
                    # Reset "original" epoch step
                    expected_current_batch = 0
                    expected_position_in_epoch = 0
                    expected_session_batch_counts = defaultdict(int)
                    expected_session_positions = {k: 0 for k in sessions}
                    expected_consumed_sessions = []
                    _update_state(
                        expected_sampler_indices,
                        _get_sampler_attributes(fast_dataloader, _attr_name='indices'),
                        _validate_change=True, # Check that indices were shuffled
                    )
                    _update_state(
                        expected_prv_rng_states,
                        _get_sampler_attributes(fast_dataloader, _attr_name='prv_rng_state'),
                        _validate_change=True, # Check that RNG state was updated
                    )
                    # Check automatic reset after exhausting iterator
                    _check_reset(fast_dataloader)
                    # Reset iterator
                    dl_iter = iter(fast_dataloader)
                    # Check that `iter` doesn't affect state
                    _check_reset(fast_dataloader)
                    # Get next batch from new iterator
                    key, batch = next(dl_iter)
                # Reset batch cycle
                if len(expected_consumed_sessions) == len(sessions):
                    expected_consumed_sessions = []
                    expected_position_in_epoch += 1
                # Reset session specific iterator
                if expected_session_positions[key] == len(fast_dataloader.session_dataloaders[key]):
                    expected_session_positions[key] = 0
                    _update_state(
                        expected_sampler_indices,
                        _get_sampler_attributes(fast_dataloader, _attr_name='indices'),
                        _session_name=key,
                        _validate_change=True, # Check that indices were shuffled
                    )
                    _update_state(
                        expected_prv_rng_states,
                        _get_sampler_attributes(fast_dataloader, _attr_name='prv_rng_state'),
                        _session_name=key,
                        _validate_change=True, # Check that RNG state was updated
                    )
                # Update expected values
                expected_current_batch += 1
                expected_session_batch_counts[key] += 1
                expected_session_positions[key] += 1
                expected_consumed_sessions.append(key)
                # Validate expected values
                #  `current_batch`
                assert expected_current_batch == fast_dataloader.current_batch
                #  `position_in_epoch`
                assert expected_position_in_epoch == fast_dataloader.position_in_epoch
                #  `batches_from_session`
                assert expected_session_batch_counts == fast_dataloader.batches_from_session
                #  `session_positions`
                assert (
                    expected_session_positions ==
                    fast_dataloader.session_positions ==
                    _get_sampler_attributes(fast_dataloader, _attr_name='position')
                )
                #  `SessionSpecificSampler.indices`
                assert expected_sampler_indices == _get_sampler_attributes(
                    fast_dataloader, _attr_name='indices'
                )
                #  `SessionSpecificSampler.prv_rng_state`
                assert states_equal(expected_prv_rng_states, _get_sampler_attributes(
                    fast_dataloader, _attr_name='prv_rng_state'
                ))
                #  `BatchSampler.consumed_sessions`
                assert expected_consumed_sessions == fast_dataloader.batch_sampler.consumed_sessions
                # Update epoch counters
                epoch_step += 1
                epoch_batches_per_session[key] += 1
                epoch_batch_hashes_per_session[key].extend(
                    [hash(x.flatten().tolist().__str__()) for x in batch['responses']]
                )
            # Validate after epoch counts
            #  total batches
            assert epoch_step == synced_len
            max_batches = fast_dataloader.max_batches_per_session
            for session_name, session_dl in fast_dataloader.session_dataloaders.items():
                batch_counts = epoch_batches_per_session[session_name]
                batch_hash_counts = Counter(epoch_batch_hashes_per_session[session_name])
                #  batches per session
                assert (
                    int(1.5 * max_batches) <= batch_counts <= math.ceil(1.5 * max_batches)
                )
                max_samples = max_batches * session_dl.batch_sampler.batch_size
                sess_len = len(session_dl) * session_dl.batch_sampler.batch_size
                sess_min = int(max_samples / sess_len) + int(.5 * max_samples / sess_len)
                sess_max = math.ceil(max_samples / sess_len) + math.ceil(.5 * max_samples / sess_len)
                #  Appearances of each unique batch
                assert all(sess_min <= count <= sess_max for count in batch_hash_counts.values() )
            # Reset dataloader
            fast_dataloader.reset_state()
            # Check that dataloader state was reset
            _check_reset(fast_dataloader)
            epochs -= 1
    
    def test_state_save_restore(self, fast_dataloader_factory):
        """Test state save and restore functionality."""
        fast_dataloader = fast_dataloader_factory(batch_size=1)
        # Iterate through part of an epoch
        original_batches = []
        iterations_before_save = 12  # Save after 12 iterations
        
        iterator = iter(fast_dataloader)
        for _ in range(iterations_before_save):
            batch = next(iterator)
            original_batches.append(batch)
        
        # Save state
        saved_state = fast_dataloader.get_state()
        
        # Create new dataloader with same configuration
        new_dataloader = fast_dataloader_factory(batch_size=1)
        
        # Restore state
        new_dataloader.set_state(saved_state)
        
        # Continue with original dataloader
        remaining_original = list(iterator)
        
        # Get remaining batches from restored dataloader
        remaining_restored = list(new_dataloader)
        
        # Should match exactly
        assert len(remaining_original) == len(remaining_restored)
        
        # Compare session names and batch data (responses tensor should be identical)
        for (orig_session, orig_data), (rest_session, rest_data) in zip(remaining_original, remaining_restored):
            assert orig_session == rest_session
            # Compare responses tensors (our deterministic mock data)
            assert torch.equal(orig_data['responses'], rest_data['responses'])
    
    def test_iter_behavior_maintains_state(self, fast_dataloader_factory):
        """Test that calling iter() multiple times doesn't reset state inappropriately."""
        fast_dataloader = fast_dataloader_factory()
        # Get initial state
        initial_state = fast_dataloader.get_state()
        
        # Call iter() multiple times
        iter1 = iter(fast_dataloader)
        batch1 = next(iter1)
        iter2 = iter(fast_dataloader)
        batch2 = next(iter2) 
        iter3 = iter(fast_dataloader)
        batch3 = next(iter3)
        
        # Should be the same batch (same session and batch data)
        assert batch1[0] != batch2[0] != batch3[0]  # Different sessions
        assert all(p == 1 for p in fast_dataloader.session_positions.values())
        assert all(dl.batch_sampler.position == 1 for dl in fast_dataloader.session_dataloaders.values())
        assert fast_dataloader.current_batch == 3
        assert all(bfs == 1 for bfs in fast_dataloader.batches_from_session.values())
        # NOTE: `position_in_epoch` is incremented only after `yield`
        assert fast_dataloader.position_in_epoch == 0
        # TODO: This actually fails because each iterator has it's own `cycle_order` generated!
        #  This behavior should be modified such that the cycle order is in the underlying state of
        #  of the batch sampler.
        # next(iter1)
        # assert fast_dataloader.position_in_epoch == 1
    
    def test_deterministic_behavior_with_seed(self, fast_dataloader_factory):
        """Test that same seed produces deterministic behavior."""
        
        # Create two dataloaders with same seed
        dl1 = fast_dataloader_factory(seed=99999)
        dl2 = fast_dataloader_factory(seed=99999)
        
        # Get first 10 batches from each
        batches1 = []
        batches2 = []
        
        for i, ((session1, data1), (session2, data2)) in enumerate(zip(dl1, dl2)):
            # Use tensor sum as a simple but deterministic hash
            responses_sum1 = data1['responses'].sum().item()
            responses_sum2 = data2['responses'].sum().item()
            batches1.append((session1, responses_sum1))
            batches2.append((session2, responses_sum2))
            if i >= 9:  # Get first 10 batches
                break
        
        # Should be identical
        assert batches1 == batches2


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""
    
    def test_partial_epoch_state_restoration(self, fast_dataloader_factory):
        """Test state restoration at various points in an epoch."""
        fast_dataloader = fast_dataloader_factory()
        checkpoints = [3, 7, 15, 28]  # Various points to test
        
        for checkpoint_at in checkpoints:
            # Reset dataloader
            fast_dataloader.reset_state()
            
            # Iterate to checkpoint
            original_batches = []
            iterator = iter(fast_dataloader)
            for _ in range(checkpoint_at):
                try:
                    original_batches.append(next(iterator))
                except StopIteration:
                    # Reset iterator
                    iterator = iter(fast_dataloader)
                    original_batches.append(next(iterator))
            
            # Save state
            state = fast_dataloader.get_state()
            
            # Continue original
            remaining_original = list(iterator)
            
            # Create fresh dataloader and restore
            new_dataloader = fast_dataloader_factory()
            new_dataloader.set_state(state)
            
            # Get remaining from restored
            remaining_restored = list(new_dataloader)
            
            # Should match
            assert len(remaining_original) == len(remaining_restored)
            for (orig_session, orig_data), (rest_session, rest_data) in zip(remaining_original, remaining_restored):
                assert orig_session == rest_session
                assert torch.equal(orig_data['responses'], rest_data['responses'])
    
    def test_state_save_at_end_of_iterator(self, fast_dataloader_factory):
        """Test state save/restore behavior when saving at the very end of iteration."""
        fast_dataloader = fast_dataloader_factory()
        
        # Consume all but the last batch
        iterator = iter(fast_dataloader)
        all_batches = []
        
        # Get all batches except the last one
        for i in range(len(fast_dataloader) - 1):
            batch = next(iterator)
            all_batches.append(batch)
        
        # Get the last batch
        last_batch = next(iterator)
        all_batches.append(last_batch)
        
        # Now we're at the end - save state here
        end_state = fast_dataloader.get_state()
        
        # Verify that trying to get next batch raises StopIteration
        with pytest.raises(StopIteration):
            next(iterator)
        
        # Create new dataloader and restore the end state
        new_dataloader = fast_dataloader_factory()
        new_dataloader.set_state(end_state)
        
        # Create iterator from restored dataloader
        restored_iterator = iter(new_dataloader)
        
        # Trying to get next batch should immediately raise StopIteration
        with pytest.raises(StopIteration):
            next(restored_iterator)
        
        # Verify that both dataloaders are in the same "exhausted" state
        assert states_equal(fast_dataloader.get_state(), new_dataloader.get_state())
        # NOTE: Dataloaders should automatically reset state after exhausting iterator
        _check_reset(fast_dataloader)
        _check_reset(new_dataloader)
        
        # Verify that after reset, both work normally
        fast_dataloader.reset_state()
        new_dataloader.reset_state()
        # NOTE: Verify that dataloader state is still reset
        _check_reset(fast_dataloader)
        _check_reset(new_dataloader)
        
        # Both should be able to iterate normally after reset
        first_batch_original = next(iter(fast_dataloader))
        first_batch_restored = next(iter(new_dataloader))
        
        # Should get the same first batch (same seed)
        assert first_batch_original[0] == first_batch_restored[0]  # Same session
        assert torch.equal(first_batch_original[1]['responses'], first_batch_restored[1]['responses'])
    
    def test_shuffle_indices_behavior(self, fast_dataloader_factory):
        """Test that shuffle_indices produces different orders but is reproducible with same RNG state."""
        fast_dataloader = fast_dataloader_factory()
        
        # Get a session sampler that uses shuffling
        session_name = "session_a"
        session_sampler = fast_dataloader.session_dataloaders[session_name].batch_sampler
        
        # Skip test if shuffle is disabled
        if not session_sampler.shuffle:
            pytest.skip("Test requires shuffle=True")
        
        # Reset to known state
        session_sampler.reset_state()
        original_rng_state = session_sampler.prv_rng_state
        
        # Get initial shuffled indices
        first_indices = session_sampler.indices.copy()
        
        # Shuffle again - should get different order
        session_sampler.shuffle_indices()
        second_indices = session_sampler.indices.copy()
        
        # Should be different (with high probability for non-trivial datasets)
        if len(first_indices) > 2:  # Only test if we have enough indices
            assert first_indices != second_indices, "Shuffling should produce different order"
        
        # Both should contain the same elements (just reordered)
        assert sorted(first_indices) == sorted(second_indices), "Shuffling should preserve all indices"
        
        # Reset RNG to original state and shuffle again
        session_sampler.rng.set_state(original_rng_state)
        session_sampler.shuffle_indices()
        reproduced_indices = session_sampler.indices.copy()
        
        # Should match the first shuffled result (reproducible)
        assert first_indices == reproduced_indices, "Same RNG state should produce same shuffle"
        
        # Test that multiple shuffle calls with same RNG state are deterministic
        session_sampler.rng.set_state(original_rng_state)
        session_sampler.shuffle_indices()
        session_sampler.rng.set_state(original_rng_state) 
        session_sampler.shuffle_indices()
        final_indices = session_sampler.indices.copy()
        
        # Should still match (each shuffle starts from original indices)
        assert first_indices == final_indices, "Multiple shuffles with same RNG should be identical"