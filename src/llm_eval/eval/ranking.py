"""Elo rating system for model comparison."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Rating:
    """Elo rating with metadata for tracking model performance."""

    rating: float = 1500.0
    """Current Elo rating (default starts at 1500)."""

    wins: int = 0
    """Number of wins in matches."""

    losses: int = 0
    """Number of losses in matches."""

    ties: int = 0
    """Number of ties in matches."""

    matches: int = 0
    """Total number of matches played."""


class EloRating:
    """Elo rating system for model rankings.

    This implementation follows the standard Elo rating algorithm with
    configurable K-factor and initial rating. Ratings are stored in memory
    and can be used to generate leaderboards.
    """

    K_FACTOR = 32
    """Standard K-factor used for rating updates."""

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: int = 32,
    ) -> None:
        """Initialize the Elo rating system.

        Args:
            initial_rating: Starting rating for new models (default: 1500).
            k_factor: K-factor for rating updates (default: 32).
        """
        # Guard clause: Validate k_factor
        if k_factor <= 0:
            raise ValueError("k_factor must be positive")

        self._initial_rating = initial_rating
        self._k_factor = k_factor
        self._ratings: Dict[str, Rating] = {}

    @property
    def k_factor(self) -> int:
        """Return the K-factor for rating updates."""
        return self._k_factor

    def get_rating(self, model_id: str) -> float:
        """Get current rating for a model.

        If the model has no existing rating, a new rating is created with
        the initial rating value.

        Args:
            model_id: Unique identifier for the model.

        Returns:
            Current Elo rating for the model.
        """
        if model_id not in self._ratings:
            self._ratings[model_id] = Rating(rating=self._initial_rating)
        return self._ratings[model_id].rating

    def update_ratings(
        self,
        winner: str,
        model_a_id: str,
        model_b_id: str,
    ) -> tuple[float, float]:
        """Update ratings after a match between two models.

        Uses the standard Elo formula to calculate new ratings based on
        the expected outcome versus actual outcome.

        Args:
            winner: Winner of the match: 'a', 'b', or 'tie'.
            model_a_id: ID of model A.
            model_b_id: ID of model B.

        Returns:
            Tuple of (new_rating_a, new_rating_b).

        Raises:
            ValueError: If winner is not valid or model IDs are missing.
        """
        # Guard clause: Validate winner
        if winner not in {"a", "b", "tie"}:
            raise ValueError(f"winner must be 'a', 'b', or 'tie', got: {winner}")

        # Guard clause: Validate model IDs
        if not model_a_id or not model_b_id:
            raise ValueError("model_a_id and model_b_id are required")

        # Guard clause: Validate models are different
        if model_a_id == model_b_id:
            raise ValueError("model_a_id and model_b_id must be different")

        rating_a = self.get_rating(model_a_id)
        rating_b = self.get_rating(model_b_id)

        # Calculate expected scores using standard Elo formula
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a

        # Determine actual scores based on winner
        if winner == "a":
            actual_a, actual_b = 1.0, 0.0
        elif winner == "b":
            actual_a, actual_b = 0.0, 1.0
        else:  # tie
            actual_a, actual_b = 0.5, 0.5

        # Calculate new ratings
        new_rating_a = rating_a + self._k_factor * (actual_a - expected_a)
        new_rating_b = rating_b + self._k_factor * (actual_b - expected_b)

        # Update stored ratings with match statistics
        existing_a = self._ratings.get(model_a_id, Rating())
        existing_b = self._ratings.get(model_b_id, Rating())

        self._ratings[model_a_id] = Rating(
            rating=new_rating_a,
            wins=existing_a.wins + (1 if winner == "a" else 0),
            losses=existing_a.losses + (1 if winner == "b" else 0),
            ties=existing_a.ties + (1 if winner == "tie" else 0),
            matches=existing_a.matches + 1,
        )

        self._ratings[model_b_id] = Rating(
            rating=new_rating_b,
            wins=existing_b.wins + (1 if winner == "b" else 0),
            losses=existing_b.losses + (1 if winner == "a" else 0),
            ties=existing_b.ties + (1 if winner == "tie" else 0),
            matches=existing_b.matches + 1,
        )

        return new_rating_a, new_rating_b

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get rankings sorted by rating in descending order.

        Returns:
            List of dictionaries containing model_id, rating, wins, losses,
            ties, matches, and win_rate for each model.
        """
        leaderboard = []
        for model_id, rating in self._ratings.items():
            win_rate = rating.wins / rating.matches if rating.matches > 0 else 0.0
            leaderboard.append({
                "model_id": model_id,
                "rating": rating.rating,
                "wins": rating.wins,
                "losses": rating.losses,
                "ties": rating.ties,
                "matches": rating.matches,
                "win_rate": win_rate,
            })

        # Sort by rating descending
        leaderboard.sort(key=lambda x: x["rating"], reverse=True)
        return leaderboard

    def get_rating_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed rating information for a specific model.

        Args:
            model_id: Unique identifier for the model.

        Returns:
            Dictionary with rating details, or None if model not found.
        """
        if model_id not in self._ratings:
            return None

        rating = self._ratings[model_id]
        win_rate = rating.wins / rating.matches if rating.matches > 0 else 0.0

        return {
            "model_id": model_id,
            "rating": rating.rating,
            "wins": rating.wins,
            "losses": rating.losses,
            "ties": rating.ties,
            "matches": rating.matches,
            "win_rate": win_rate,
        }

    def reset(self) -> None:
        """Reset all ratings to initial state."""
        self._ratings.clear()

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the rating system.

        Args:
            model_id: Unique identifier for the model to remove.

        Returns:
            True if model was removed, False if not found.
        """
        if model_id in self._ratings:
            del self._ratings[model_id]
            return True
        return False
