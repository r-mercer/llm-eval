"""Tests for Elo rating system."""

import pytest
from llm_eval.eval.ranking import EloRating, Rating


class TestEloRating:
    """Test suite for EloRating class."""

    def test_initialization(self) -> None:
        """Test EloRating initializes with correct defaults."""
        elo = EloRating()
        assert elo.k_factor == 32
        assert elo.get_rating("model1") == 1500.0

    def test_custom_initial_values(self) -> None:
        """Test EloRating with custom initial values."""
        elo = EloRating(initial_rating=1200, k_factor=16)
        assert elo.get_rating("model1") == 1200.0
        assert elo.k_factor == 16

    def test_update_ratings_winner_a(self) -> None:
        """Test rating update when model A wins."""
        elo = EloRating()
        new_a, new_b = elo.update_ratings(
            winner="a",
            model_a_id="model1",
            model_b_id="model2",
        )
        assert new_a > 1500.0
        assert new_b < 1500.0

    def test_update_ratings_winner_b(self) -> None:
        """Test rating update when model B wins."""
        elo = EloRating()
        new_a, new_b = elo.update_ratings(
            winner="b",
            model_a_id="model1",
            model_b_id="model2",
        )
        assert new_a < 1500.0
        assert new_b > 1500.0

    def test_update_ratings_tie(self) -> None:
        """Test rating update when match is a tie."""
        elo = EloRating()
        new_a, new_b = elo.update_ratings(
            winner="tie",
            model_a_id="model1",
            model_b_id="model2",
        )
        assert new_a == 1500.0
        assert new_b == 1500.0

    def test_invalid_winner_raises(self) -> None:
        """Test that invalid winner raises ValueError."""
        elo = EloRating()
        with pytest.raises(ValueError, match="winner must be"):
            elo.update_ratings(
                winner="invalid",
                model_a_id="model1",
                model_b_id="model2",
            )

    def test_same_model_raises(self) -> None:
        """Test that comparing model to itself raises ValueError."""
        elo = EloRating()
        with pytest.raises(ValueError, match="must be different"):
            elo.update_ratings(
                winner="a",
                model_a_id="model1",
                model_b_id="model1",
            )

    def test_get_leaderboard_sorted(self) -> None:
        """Test leaderboard returns models sorted by rating."""
        elo = EloRating()

        # Model 1 wins multiple times
        for _ in range(5):
            elo.update_ratings("a", "model1", "model2")

        # Model 3 wins once
        elo.update_ratings("b", "model1", "model3")

        leaderboard = elo.get_leaderboard()

        assert len(leaderboard) == 3
        assert leaderboard[0]["model_id"] == "model1"
        assert leaderboard[0]["rating"] > leaderboard[1]["rating"]

    def test_get_rating_info(self) -> None:
        """Test getting detailed rating information."""
        elo = EloRating()
        elo.update_ratings("a", "model1", "model2")

        info = elo.get_rating_info("model1")
        assert info is not None
        assert info["wins"] == 1
        assert info["losses"] == 0
        assert info["ties"] == 0
        assert info["matches"] == 1
        assert info["win_rate"] == 1.0

    def test_get_rating_info_not_found(self) -> None:
        """Test getting info for non-existent model returns None."""
        elo = EloRating()
        assert elo.get_rating_info("nonexistent") is None

    def test_reset(self) -> None:
        """Test resetting all ratings."""
        elo = EloRating()
        elo.update_ratings("a", "model1", "model2")
        elo.reset()

        assert elo.get_rating_info("model1") is None
        assert elo.get_rating_info("model2") is None

    def test_remove_model(self) -> None:
        """Test removing a model from rating system."""
        elo = EloRating()
        elo.update_ratings("a", "model1", "model2")

        assert elo.remove_model("model1") is True
        assert elo.get_rating_info("model1") is None

        # Other model should still exist
        assert elo.get_rating_info("model2") is not None

    def test_remove_model_not_found(self) -> None:
        """Test removing non-existent model returns False."""
        elo = EloRating()
        assert elo.remove_model("nonexistent") is False


class TestRating:
    """Test suite for Rating dataclass."""

    def test_default_values(self) -> None:
        """Test Rating initializes with correct defaults."""
        rating = Rating()
        assert rating.rating == 1500.0
        assert rating.wins == 0
        assert rating.losses == 0
        assert rating.ties == 0
        assert rating.matches == 0

    def test_custom_values(self) -> None:
        """Test Rating with custom values."""
        rating = Rating(rating=1600, wins=5, losses=2, ties=1, matches=8)
        assert rating.rating == 1600
        assert rating.wins == 5
        assert rating.losses == 2
        assert rating.ties == 1
        assert rating.matches == 8
