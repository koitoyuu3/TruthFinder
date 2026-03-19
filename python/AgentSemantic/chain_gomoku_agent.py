#!/usr/bin/env python3
"""
Prototype for a blockchain-connected Gomoku assistant.

This script does NOT spin up a real chain, but it demonstrates how an agent
could suggest the next move and then submit it to an oracle/ledger. The agent
can later be swapped to an LLM-based planner, and the ledger replaced with
real smart contract interactions.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

BOARD_SIZE = 15
WIN_LENGTH = 5


@dataclass
class Move:
    row: int
    col: int
    player: str  # "X" or "O"


class GomokuBoard:
    def __init__(self, size: int = BOARD_SIZE) -> None:
        self.size = size
        self.grid: List[List[str]] = [["." for _ in range(size)] for _ in range(size)]

    def apply(self, move: Move) -> bool:
        if not (0 <= move.row < self.size and 0 <= move.col < self.size):
            return False
        if self.grid[move.row][move.col] != ".":
            return False
        self.grid[move.row][move.col] = move.player
        return True

    def available_moves(self) -> List[Tuple[int, int]]:
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == ".":
                    moves.append((r, c))
        return moves

    def check_win(self, player: str) -> bool:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] != player:
                    continue
                for dr, dc in directions:
                    count = 0
                    rr, cc = r, c
                    while 0 <= rr < self.size and 0 <= cc < self.size and self.grid[rr][cc] == player:
                        count += 1
                        if count == WIN_LENGTH:
                            return True
                        rr += dr
                        cc += dc
        return False

    def to_dict(self) -> dict:
        return {"size": self.size, "grid": self.grid}

    @classmethod
    def from_dict(cls, payload: dict) -> "GomokuBoard":
        board = cls(size=payload.get("size", BOARD_SIZE))
        board.grid = payload.get("grid", board.grid)
        return board

    def __str__(self) -> str:
        rows = [" ".join(row) for row in self.grid]
        header = "   " + " ".join(f"{i:2d}" for i in range(self.size))
        body = [f"{i:2d} {rows[i]}" for i in range(self.size)]
        return "\n".join([header] + body)


class SimpleAgent:
    """
    Minimal heuristic agent: tries to place near existing stones,
    otherwise picks a random free cell. Swap this with LLM logic later.
    """

    def __init__(self, player: str) -> None:
        self.player = player

    def choose_move(self, board: GomokuBoard) -> Optional[Move]:
        moves = board.available_moves()
        if not moves:
            return None

        # Basic heuristic: prioritize positions adjacent to existing stones.
        scored: List[Tuple[int, Tuple[int, int]]] = []
        for r, c in moves:
            score = self._adjacency_score(board, r, c)
            scored.append((score, (r, c)))
        scored.sort(reverse=True, key=lambda item: item[0])

        top_score = scored[0][0]
        best_candidates = [pos for score, pos in scored if score == top_score]
        row, col = random.choice(best_candidates)
        return Move(row=row, col=col, player=self.player)

    @staticmethod
    def _adjacency_score(board: GomokuBoard, row: int, col: int) -> int:
        score = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = row + dr, col + dc
                if 0 <= rr < board.size and 0 <= cc < board.size:
                    if board.grid[rr][cc] != ".":
                        score += 1
        return score


class MockOracleLedger:
    """
    Simulated blockchain oracle that records moves.
    Replace with actual smart contract calls in production.
    """

    def __init__(self) -> None:
        self.entries: List[dict] = []

    def submit_move(self, match_id: str, move: Move) -> dict:
        payload = {
            "match_id": match_id,
            "player": move.player,
            "row": move.row,
            "col": move.col,
            "timestamp": int(time.time()),
        }
        payload["tx_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        self.entries.append(payload)
        return payload

    def dump(self) -> List[dict]:
        return list(self.entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gomoku agent with mock blockchain oracle.")
    parser.add_argument("--state", type=Path, help="Existing board JSON to resume a match.")
    parser.add_argument("--save", type=Path, default=Path("gomoku_state.json"), help="Path to save updated board.")
    parser.add_argument("--player", choices=["X", "O"], default="X", help="Agent's stone type.")
    parser.add_argument("--match-id", type=str, default="gomoku-demo", help="Identifier for the on-chain match.")
    return parser.parse_args()


def load_board(path: Optional[Path]) -> GomokuBoard:
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return GomokuBoard.from_dict(data)
    return GomokuBoard()


def save_board(board: GomokuBoard, path: Path) -> None:
    path.write_text(json.dumps(board.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    board = load_board(args.state)
    agent = SimpleAgent(player=args.player)
    oracle = MockOracleLedger()

    move = agent.choose_move(board)
    if move is None:
        print("Board is full; no move available.")
        return
    if not board.apply(move):
        print("Agent selected an invalid move; aborting.")
        return

    tx = oracle.submit_move(args.match_id, move)
    save_board(board, args.save)

    print("Agent suggested move:", move)
    print("Updated board:")
    print(board)
    print("Mock on-chain transaction:", tx)


if __name__ == "__main__":
    main()
