"""Kalman concept tracker with physics-inspired velocity constraints"""

import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from backend.models import Paper
from backend.config import settings

logger = logging.getLogger(__name__)


class KalmanConceptTracker:
    """Track concept evolution with velocity and acceleration constraints"""

    def __init__(self):
        """Initialize Kalman tracker with constraints from config"""
        # State vectors
        self.position: np.ndarray | None = None  # Current concept vector
        self.velocity: np.ndarray | None = None  # Rate of change

        # Physics constraints from config
        self.max_velocity = settings.max_velocity
        self.max_acceleration = settings.max_acceleration

        # Thresholds from config
        self.thresholds = {
            'auto_include': settings.threshold_auto_include,
            'strong': settings.threshold_strong,
            'moderate': settings.threshold_moderate,
            'reject': settings.threshold_reject
        }

        logger.info(f"Kalman tracker initialized: max_velocity={self.max_velocity}, max_acceleration={self.max_acceleration}")
        logger.info(f"Thresholds: auto={self.thresholds['auto_include']}, strong={self.thresholds['strong']}, moderate={self.thresholds['moderate']}")

    def initialize(self, seed_papers: list[Paper]) -> None:
        """
        Initialize tracker with seed papers

        Args:
            seed_papers: List of seed papers with embeddings
        """
        if not seed_papers:
            raise ValueError("Cannot initialize with empty seed papers")

        if any(p.embedding is None for p in seed_papers):
            raise ValueError("All seed papers must have embeddings")

        # Convert embeddings to numpy array
        embeddings = np.array([p.embedding for p in seed_papers])

        # Initial position = mean of seed embeddings
        self.position = embeddings.mean(axis=0)
        self.position /= np.linalg.norm(self.position)  # Normalize

        # Zero initial velocity
        self.velocity = np.zeros_like(self.position)

        logger.info(f"Initialized with {len(seed_papers)} seeds")
        logger.info(f"Position norm: {np.linalg.norm(self.position):.3f}")
        logger.debug(f"Position shape: {self.position.shape}")

    def evaluate_candidate(self, candidate_vector: np.ndarray) -> tuple[bool, float, str]:
        """
        Check if candidate satisfies Kalman constraints

        Args:
            candidate_vector: Embedding vector of candidate paper

        Returns:
            Tuple of (is_valid, confidence, reason)
        """
        if self.position is None:
            raise ValueError("Tracker not initialized")

        # Ensure candidate is normalized
        candidate_vector = candidate_vector / np.linalg.norm(candidate_vector)

        # 1. Similarity to current position
        similarity = cosine_similarity(
            [self.position],
            [candidate_vector]
        )[0][0]

        if similarity < self.thresholds['reject']:
            return False, 0.0, f"Similarity {similarity:.3f} below threshold {self.thresholds['reject']}"

        # 2. Implied velocity check
        implied_velocity = candidate_vector - self.position
        velocity_magnitude = np.linalg.norm(implied_velocity)

        if velocity_magnitude > self.max_velocity:
            return False, 0.0, f"Velocity {velocity_magnitude:.4f} exceeds max {self.max_velocity}"

        # 3. Acceleration check (change in velocity)
        velocity_change = implied_velocity - self.velocity
        acceleration = np.linalg.norm(velocity_change)

        if acceleration > self.max_acceleration:
            return False, 0.0, f"Acceleration {acceleration:.4f} exceeds max {self.max_acceleration}"

        # 4. Calculate confidence based on similarity
        if similarity >= self.thresholds['auto_include']:
            confidence = 0.95
            tier = "auto"
        elif similarity >= self.thresholds['strong']:
            confidence = 0.80
            tier = "strong"
        elif similarity >= self.thresholds['moderate']:
            confidence = 0.65
            tier = "moderate"
        else:
            confidence = 0.50
            tier = "weak"

        # Reduce confidence for high acceleration
        acceleration_penalty = acceleration / self.max_acceleration
        confidence *= (1.0 - 0.3 * acceleration_penalty)

        logger.debug(f"Accepted ({tier}): sim={similarity:.3f}, vel={velocity_magnitude:.4f}, acc={acceleration:.4f}, conf={confidence:.3f}")

        return True, confidence, f"Passes all constraints ({tier} similarity)"

    def process_papers(self, papers: list[Paper]) -> tuple[list[tuple], list[tuple]]:
        """
        Evaluate all papers in window

        Args:
            papers: List of papers with embeddings to evaluate

        Returns:
            Tuple of (accepted_papers, rejected_papers)
            Each entry is (paper, similarity, confidence, reason)
        """
        if self.position is None:
            raise ValueError("Tracker not initialized")

        accepted = []
        rejected = []

        for paper in papers:
            if paper.embedding is None:
                logger.warning(f"Skipping paper {paper.arxiv_id} - no embedding")
                continue

            # Convert to numpy array and normalize
            embedding = np.array(paper.embedding)
            embedding = embedding / np.linalg.norm(embedding)

            # Calculate similarity
            similarity = cosine_similarity(
                [self.position],
                [embedding]
            )[0][0]

            # Quick reject if below threshold
            if similarity < self.thresholds['reject']:
                rejected.append((paper, similarity, 0.0, "Below similarity threshold"))
                continue

            # Evaluate with Kalman constraints
            is_valid, confidence, reason = self.evaluate_candidate(embedding)

            if is_valid:
                accepted.append((paper, similarity, confidence, reason))
            else:
                rejected.append((paper, similarity, confidence, reason))

        logger.info(f"Processed {len(papers)} papers: {len(accepted)} accepted, {len(rejected)} rejected")

        return accepted, rejected

    def update(self, accepted_papers: list[tuple]) -> None:
        """
        Update tracker state with accepted papers (Kalman-style update)

        Args:
            accepted_papers: List of (paper, similarity, confidence, reason) tuples
        """
        if not accepted_papers:
            logger.warning("No papers to update with")
            return

        if self.position is None:
            raise ValueError("Tracker not initialized")

        # Extract embeddings and confidences
        embeddings = []
        confidences = []

        for paper, similarity, conf, reason in accepted_papers:
            embedding = np.array(paper.embedding)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
            confidences.append(conf)

        embeddings = np.array(embeddings)
        confidences = np.array(confidences)

        # Weighted mean of accepted papers
        weights = confidences / confidences.sum()
        new_position = (embeddings.T @ weights)
        new_position = new_position / np.linalg.norm(new_position)

        # Kalman-style update (blend new measurement with current position)
        kalman_gain = 0.3  # Blend factor
        innovation = new_position - self.position

        self.position = self.position + kalman_gain * innovation
        self.position = self.position / np.linalg.norm(self.position)

        # Update velocity (smoothed exponential moving average)
        new_velocity = innovation
        self.velocity = 0.8 * self.velocity + 0.2 * new_velocity

        # Clip velocity to maximum
        velocity_mag = np.linalg.norm(self.velocity)
        if velocity_mag > self.max_velocity:
            self.velocity *= (self.max_velocity / velocity_mag)
            logger.debug(f"Clipped velocity from {velocity_mag:.4f} to {self.max_velocity}")

        logger.info(f"Updated tracker: velocity_mag={np.linalg.norm(self.velocity):.4f}, num_papers={len(accepted_papers)}")
        logger.debug(f"Position drift: {np.linalg.norm(innovation):.4f}")
