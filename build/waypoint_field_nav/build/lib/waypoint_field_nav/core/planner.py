"""Global planner — step 3b of the plan: a Dijkstra search weighted by
distance-from-goal.

Plain Dijkstra expands by accumulated path cost g(n). The plan asks for a
"modified Dijkstra weighted based on distance from goal", which is exactly
weighted A*:

    f(n) = g(n) + w * h(n),   h = straight-line distance to goal

* w = 0  -> pure Dijkstra (uniform-cost search)
* w = 1  -> standard A* (optimal)
* w > 1  -> greedy toward the goal, expands far fewer nodes (default here)

The grid is built once from the obstacle list; cells whose centre lies within
(obstacle surface + inflation) are marked blocked, so the returned path already
keeps the robot's body clear. 8-connected with proper diagonal costs.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry


class OccupancyGrid:
    def __init__(
        self,
        bounds: Tuple[float, float, float, float],  # (min_x, min_y, max_x, max_y)
        resolution: float,
        obstacles: Sequence[Dict],
        inflation: float,
    ) -> None:
        self.min_x, self.min_y, self.max_x, self.max_y = bounds
        self.res = resolution
        self.inflation = inflation
        self.nx = max(1, int(math.ceil((self.max_x - self.min_x) / resolution)))
        self.ny = max(1, int(math.ceil((self.max_y - self.min_y) / resolution)))
        self.blocked = np.zeros((self.nx, self.ny), dtype=bool)
        for ix in range(self.nx):
            for iy in range(self.ny):
                wx, wy = self.cell_to_world(ix, iy)
                if geometry.point_blocked(wx, wy, obstacles, inflation):
                    self.blocked[ix, iy] = True

    # --- coordinate helpers ------------------------------------------------
    def world_to_cell(self, wx: float, wy: float) -> Tuple[int, int]:
        ix = int((wx - self.min_x) / self.res)
        iy = int((wy - self.min_y) / self.res)
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        return ix, iy

    def cell_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        return (self.min_x + (ix + 0.5) * self.res,
                self.min_y + (iy + 0.5) * self.res)

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def is_free(self, ix: int, iy: int) -> bool:
        return self.in_bounds(ix, iy) and not self.blocked[ix, iy]

    def nearest_free(self, ix: int, iy: int, max_r: int = 6) -> Optional[Tuple[int, int]]:
        """If a cell is blocked (e.g. the goal sits in inflation), find the
        closest free cell by ring search so a plan can still be produced."""
        if self.is_free(ix, iy):
            return ix, iy
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    if self.is_free(ix + dx, iy + dy):
                        return ix + dx, iy + dy
        return None

    def reachable_mask(self, start_cell: Tuple[int, int]) -> np.ndarray:
        """Boolean mask of the free cells reachable from ``start_cell`` by
        8-connected moves, using the SAME no-diagonal-corner-cutting rule as
        ``plan()``. This is the connectivity used by the pre-flight
        waypoint-reachability check: a waypoint whose (snapped) cell is not in
        this mask is walled off from the start and can never be driven to.
        """
        mask = np.zeros((self.nx, self.ny), dtype=bool)
        sx, sy = start_cell
        if not self.is_free(sx, sy):
            return mask
        mask[sx, sy] = True
        dq = deque([start_cell])
        while dq:
            cx, cy = dq.popleft()
            for dx, dy, _ in _NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                if not self.is_free(nx, ny) or mask[nx, ny]:
                    continue
                if dx != 0 and dy != 0:  # don't squeeze through a diagonal gap
                    if self.blocked[cx + dx, cy] and self.blocked[cx, cy + dy]:
                        continue
                mask[nx, ny] = True
                dq.append((nx, ny))
        return mask


_NEIGHBORS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


def plan(
    grid: OccupancyGrid,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    goal_weight: float = 1.6,
) -> Optional[List[Tuple[float, float]]]:
    """Weighted-A* / Dijkstra search. Returns a list of world waypoints from
    start to goal (inclusive), or None if unreachable."""
    start = grid.nearest_free(*grid.world_to_cell(*start_xy))
    goal = grid.nearest_free(*grid.world_to_cell(*goal_xy))
    if start is None or goal is None:
        return None

    def h(cell: Tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    open_heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
    g_cost: Dict[Tuple[int, int], float] = {start: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed = set()

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            return _reconstruct(grid, came_from, cur, goal_xy)
        closed.add(cur)
        cx, cy = cur
        for dx, dy, step in _NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not grid.is_free(nx, ny):
                continue
            # Don't cut diagonal corners between two blocked cells.
            if dx != 0 and dy != 0:
                if grid.blocked[cx + dx, cy] and grid.blocked[cx, cy + dy]:
                    continue
            ng = g_cost[cur] + step
            nb = (nx, ny)
            if ng < g_cost.get(nb, math.inf):
                g_cost[nb] = ng
                came_from[nb] = cur
                f = ng + goal_weight * h(nb)      # <-- distance-from-goal weighting
                heapq.heappush(open_heap, (f, nb))
    return None


def _reconstruct(grid, came_from, cur, goal_xy) -> List[Tuple[float, float]]:
    cells = [cur]
    while cur in came_from:
        cur = came_from[cur]
        cells.append(cur)
    cells.reverse()
    path = [grid.cell_to_world(ix, iy) for ix, iy in cells]
    path.append((goal_xy[0], goal_xy[1]))     # end exactly on the requested goal
    return path
