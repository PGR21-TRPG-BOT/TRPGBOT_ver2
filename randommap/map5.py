import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import heapq
from collections import deque
import itertools

# 폰트 설정 (한글 표시)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 새로운 기능 관련 상수 =====
FEATURE_NONE = 0
FEATURE_TRAP = 1
FEATURE_SECRET_HINT = 2
FEATURE_OBSTACLE = 3
FEATURE_BRIDGE = 4
FEATURE_ENTRANCE = 5
FEATURE_EXIT = 6
FEATURE_PATH = 7

# ==============================

# 던전과 높이 맵 생성 함수
def generate_dungeon(width, height, room_count=8, room_min=8, room_max=15, min_room_distance=4, corridor_width_options=[1, 2], extra_connection_prob=0.3):
    dungeon = np.zeros((height, width), dtype=int)
    rooms = []
    attempts = 0
    max_attempts = 200

    # 방 생성
    while len(rooms) < room_count and attempts < max_attempts:
        w = np.random.randint(room_min, room_max)
        h = np.random.randint(room_min, room_max)
        x = np.random.randint(1, width - w - 1)
        y = np.random.randint(1, height - h - 1)
        new_room_rect = (x, y, w, h)
        
        too_close = False
        for rx, ry, rw, rh in rooms:
            if not (x + w + min_room_distance <= rx or rx + rw + min_room_distance <= x or
                   y + h + min_room_distance <= ry or ry + rh + min_room_distance <= y):
                too_close = True
                break
        
        attempts += 1
        if too_close: continue
            
        dungeon[y:y+h, x:x+w] = 1
        rooms.append(new_room_rect)

    if len(rooms) < 2:
        print("방을 충분히 생성하지 못했습니다.")
        return dungeon, rooms, [] # 빈 복도 리스트 반환

    # 방 연결 (MST + 추가 연결)
    connected = {0}
    edges = []
    room_centers = [(r[0] + r[2]//2, r[1] + r[3]//2) for r in rooms]

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            dist = abs(room_centers[i][0] - room_centers[j][0]) + abs(room_centers[i][1] - room_centers[j][1])
            edges.append((dist, i, j))
    
    edges.sort()
    
    corridors = [] # 복도 타일 좌표 리스트들의 리스트
    mst_edges_count = 0

    # MST 연결
    for dist, i, j in edges:
        if len(connected) == len(rooms): break # 이미 모든 방이 연결됨
        if i in connected and j not in connected or i not in connected and j in connected:
            connected.add(i); connected.add(j)
            x1, y1 = room_centers[i]; x2, y2 = room_centers[j]
            corridor_width = np.random.choice(corridor_width_options)
            points = create_corridor(dungeon, x1, y1, x2, y2, corridor_width)
            if points: corridors.append(points)
            mst_edges_count += 1

    # 추가 연결 (곁가지)
    remaining_edges = edges[mst_edges_count:] # MST에 사용되지 않은 엣지
    np.random.shuffle(remaining_edges) # 무작위로 섞음
    
    num_extra_connections = int(len(remaining_edges) * extra_connection_prob) # 추가할 연결 수
    
    for k in range(min(num_extra_connections, len(remaining_edges))):
        dist, i, j = remaining_edges[k]
        x1, y1 = room_centers[i]; x2, y2 = room_centers[j]
        # 너무 짧은 거리는 곁가지로 잘 추가하지 않음 (옵션)
        # if dist < (room_min + room_max) / 2: continue 
        
        print(f"곁가지 복도 시도: 방 {i} <-> 방 {j}")
        corridor_width = np.random.choice(corridor_width_options)
        points = create_corridor(dungeon, x1, y1, x2, y2, corridor_width)
        if points: corridors.append(points)

    return dungeon, rooms, corridors


def create_corridor(dungeon, x1, y1, x2, y2, width):
    """두 점 사이에 L자 또는 Z자 복도를 생성하고 타일 좌표 리스트 반환"""
    height, map_width = dungeon.shape
    points = []
    
    # 복도 생성 (L자 또는 Z자)
    if np.random.rand() < 0.7: # L자
        if np.random.rand() < 0.5: # 수직->수평
            for cy in range(min(y1, y2), max(y1, y2) + 1):
                for offset in range(width):
                    px = x1 + offset
                    if 0 <= px < map_width and 0 <= cy < height: dungeon[cy, px] = 1; points.append((cy, px))
            for cx in range(min(x1, x2), max(x1, x2) + 1):
                 for offset in range(width):
                    py = y2 + offset
                    if 0 <= py < height and 0 <= cx < map_width: dungeon[py, cx] = 1; points.append((py, cx))
        else: # 수평->수직
            for cx in range(min(x1, x2), max(x1, x2) + 1):
                for offset in range(width):
                    py = y1 + offset
                    if 0 <= py < height and 0 <= cx < map_width: dungeon[py, cx] = 1; points.append((py, cx))
            for cy in range(min(y1, y2), max(y1, y2) + 1):
                 for offset in range(width):
                    px = x2 + offset
                    if 0 <= px < map_width and 0 <= cy < height: dungeon[cy, px] = 1; points.append((cy, px))
    else: # Z자
        mid_x = np.random.randint(min(x1, x2), max(x1, x2) + 1) if x1 != x2 else x1
        mid_y = np.random.randint(min(y1, y2), max(y1, y2) + 1) if y1 != y2 else y1
        # y1 -> mid_y (수직)
        for cy in range(min(y1, mid_y), max(y1, mid_y) + 1):
            for offset in range(width):
                px = x1 + offset
                if 0 <= px < map_width and 0 <= cy < height: dungeon[cy, px] = 1; points.append((cy, px))
        # x1 -> mid_x (수평, mid_y)
        for cx in range(min(x1, mid_x), max(x1, mid_x) + 1):
            for offset in range(width):
                py = mid_y + offset
                if 0 <= py < height and 0 <= cx < map_width: dungeon[py, cx] = 1; points.append((py, cx))
        # mid_y -> y2 (수직, mid_x)
        for cy in range(min(mid_y, y2), max(mid_y, y2) + 1):
            for offset in range(width):
                px = mid_x + offset
                if 0 <= px < map_width and 0 <= cy < height: dungeon[cy, px] = 1; points.append((cy, px))
        # mid_x -> x2 (수평, y2)
        for cx in range(min(mid_x, x2), max(mid_x, x2) + 1):
             for offset in range(width):
                py = y2 + offset
                if 0 <= py < height and 0 <= cx < map_width: dungeon[py, cx] = 1; points.append((py, cx))

    # 중복 제거 후 반환
    return list(set(points))


def generate_height_map(dungeon, rooms, corridors, smoothness=3, max_height=15, corridor_height_range=(1, 4), obstacle_prob=0.05, obstacle_height_range=(1, 3)):
    height, width = dungeon.shape
    noise = np.random.rand(height, width)
    
    for _ in range(smoothness):
        noise = (noise + np.roll(noise, 1, axis=0) + np.roll(noise, -1, axis=0) +
                 np.roll(noise, 1, axis=1) + np.roll(noise, -1, axis=1)) / 5
    
    height_map = (noise * max_height).astype(int) * (dungeon == 1)
    feature_map = np.zeros_like(dungeon, dtype=int) # ===== 지형 특성 맵 추가 =====
    
    room_heights = np.random.randint(1, max_height + 1, size=len(rooms))
    
    # 방 내부 높이 설정
    for i, (x, y, w, h) in enumerate(rooms):
        base_height = room_heights[i]
        for ry in range(y, y + h):
            for rx in range(x, x + w):
                if dungeon[ry, rx] == 1:
                    is_edge = rx == x or rx == x + w - 1 or ry == y or ry == y + h - 1
                    height_map[ry, rx] = max(1, base_height + (np.random.randint(-1, 2) if is_edge else np.random.randint(-1, 1)))

    # 복도 높이 및 특성 설정
    all_corridor_points = set(itertools.chain(*corridors)) if corridors else set()
    processed_corridor_points = set()

    for r in range(height):
        for c in range(width):
            if dungeon[r, c] == 1 and (r, c) in all_corridor_points:
                 # 복도 타일만 처리 (방 내부는 제외)
                 is_in_room_interior = False
                 for rx, ry, rw, rh in rooms:
                     if rx < c < rx + rw -1 and ry < r < ry + rh -1:
                         is_in_room_interior = True
                         break
                 if not is_in_room_interior:
                    # 복도 기본 높이
                    if height_map[r, c] == 0: # 아직 높이 할당 안된 경우 (복도가 방 위에 생성된 경우 등)
                       height_map[r, c] = np.random.randint(corridor_height_range[0], corridor_height_range[1] + 1)
                    else: # 기존 높이 유지 또는 약간 조정
                       height_map[r, c] = max(corridor_height_range[0], min(corridor_height_range[1], height_map[r, c]))

                    # 복도 특성 부여
                    rand_val = np.random.rand()
                    if rand_val < obstacle_prob: # 장애물
                        # 주변이 벽이 아닌 복도인지 확인
                         is_surrounded_by_corridor = True
                         neighbor_coords = []
                         for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                             nr, nc = r + dr, c + dc
                             if not (0 <= nr < height and 0 <= nc < width and dungeon[nr, nc] == 1 and (nr, nc) in all_corridor_points):
                                 is_surrounded_by_corridor = False; break
                             neighbor_coords.append((nr, nc))
                         
                         if is_surrounded_by_corridor:
                            height_map[r, c] += np.random.randint(obstacle_height_range[0], obstacle_height_range[1] + 1)
                            feature_map[r, c] = FEATURE_OBSTACLE
                            print(f"복도 장애물 추가: ({r}, {c}), 높이: {height_map[r, c]}")
                            
                    # ===== 다리(Bridge) 지형 감지 =====
                    # 좁은 복도(너비 1)이고 아래가 비어있거나 매우 낮은 경우
                    elif dungeon[r, c] == 1: # 복도 타일이고
                        is_narrow_bridge_candidate = True
                        # 좌우 또는 상하 중 하나는 벽이어야 좁은 복도로 간주
                        is_horizontal_corridor = (dungeon[r, c-1] == 0 or dungeon[r, c+1] == 0) if 0 < c < width - 1 else True
                        is_vertical_corridor = (dungeon[r-1, c] == 0 or dungeon[r+1, c] == 0) if 0 < r < height - 1 else True
                        
                        if not (is_horizontal_corridor ^ is_vertical_corridor): # 정확히 한 방향으로만 벽이 있어야 함 (너비 1 복도 추정)
                             is_narrow_bridge_candidate = False

                        if is_narrow_bridge_candidate:
                             # 아래 방향 확인 (아이소메트릭 기준 아래는 y+1, x+1 또는 y+1, x-1 쪽)
                             # 여기서는 단순화: 바로 아래(y+1) 타일이 비어있거나 높이 차이가 크면 다리로 간주
                             if r + 1 < height and dungeon[r+1, c] == 0: # 바로 아래가 비어있음
                                 feature_map[r, c] = FEATURE_BRIDGE
                                 print(f"다리 지형 감지: ({r}, {c})")
                             elif r + 1 < height and dungeon[r+1, c] == 1: # 아래 타일이 있지만 높이 차이가 클 때
                                 if height_map[r, c] - height_map[r+1, c] > 5: # 예: 5 이상 차이
                                     feature_map[r, c] = FEATURE_BRIDGE
                                     print(f"높이차 다리 지형 감지: ({r}, {c})")
                                     
                    processed_corridor_points.add((r, c))


    # ===== 함정(Trap) 및 비밀(Secret Hint) 배치 =====
    # 함정은 복도나 방 가장자리에 배치
    # 비밀 힌트는 막다른 복도 끝에 배치
    
    trap_prob = 0.02 # 함정 배치 확률
    secret_hint_prob = 0.1 # 비밀 힌트 배치 확률 (막다른 길 기준)
    
    all_dungeon_points = np.argwhere(dungeon == 1)
    
    for y, x in all_dungeon_points:
        # 이미 다른 특성이 할당된 곳은 제외
        if feature_map[y, x] != FEATURE_NONE: continue 
            
        # 주변 타일 개수 확인
        live_neighbors = 0
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
             ny, nx = y + dy, x + dx
             if 0 <= ny < height and 0 <= nx < width and dungeon[ny, nx] == 1:
                 live_neighbors += 1
                 neighbors.append((ny, nx))

        # 비밀 힌트: 막다른 길 (이웃이 1개)
        if live_neighbors == 1:
             if np.random.rand() < secret_hint_prob:
                 feature_map[y, x] = FEATURE_SECRET_HINT
                 print(f"비밀 힌트 추가: ({y}, {x})")
        # 함정: 복도 중간(이웃 2개) 또는 방 가장자리(이웃 3개 이상, 벽 근처)
        elif live_neighbors >= 2 : 
             is_on_edge = False # 방 또는 복도의 가장자리인가?
             for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                  ny, nx = y + dy, x + dx
                  if not (0 <= ny < height and 0 <= nx < width and dungeon[ny, nx] == 1):
                       is_on_edge = True; break
             
             # 복도 중간이거나 가장자리에 함정 배치 가능
             if is_on_edge or live_neighbors == 2:
                  if np.random.rand() < trap_prob:
                      feature_map[y, x] = FEATURE_TRAP
                      print(f"함정 추가: ({y}, {x})")

    # 높이 최소값 보정
    height_map = np.maximum(height_map, 1) * (dungeon == 1)
    
    return height_map, feature_map # ===== 특성 맵 반환 =====


def select_entrance_exit(dungeon, rooms, height_map, feature_map):
    """입구와 출구 선택, 출구 주변 높이 조정 및 특성 맵 업데이트"""
    if not rooms: return None, None
        
    entrance_room_idx = np.random.randint(0, len(rooms))
    if len(rooms) == 1 or np.random.rand() < 0.2:
        exit_room_idx = entrance_room_idx
    else:
        possible_exit_rooms = list(range(len(rooms)))
        possible_exit_rooms.remove(entrance_room_idx)
        exit_room_idx = np.random.choice(possible_exit_rooms)
    
    entrance_room = rooms[entrance_room_idx]
    exit_room = rooms[exit_room_idx]
    
    ex, ey, ew, eh = entrance_room
    entrance_x = np.random.randint(ex + 1, ex + ew - 1) if ew > 2 else ex
    entrance_y = np.random.randint(ey + 1, ey + eh - 1) if eh > 2 else ey
    
    xx, xy, xw, xh = exit_room
    exit_x = np.random.randint(xx + 1, xx + xw - 1) if xw > 2 else xx
    exit_y = np.random.randint(xy + 1, xy + xh - 1) if xh > 2 else xy

    while entrance_room_idx == exit_room_idx and entrance_x == exit_x and entrance_y == exit_y:
        if xw > 2 : exit_x = np.random.randint(xx + 1, xx + xw - 1)
        if xh > 2 : exit_y = np.random.randint(xy + 1, xy + xh - 1)
        
    entrance = (entrance_y, entrance_x)
    exit_coords = (exit_y, exit_x)

    # 입구/출구 타일 높이 설정 및 특성 부여
    height_map[entrance[0], entrance[1]] = max(1, height_map[entrance[0], entrance[1]]) # 0이 되지 않도록
    height_map[exit_coords[0], exit_coords[1]] = 1 # 출구는 낮게
    
    feature_map[entrance[0], entrance[1]] = FEATURE_ENTRANCE
    feature_map[exit_coords[0], exit_coords[1]] = FEATURE_EXIT
    
    print(f"입구 ({entrance[0]}, {entrance[1]}), 출구 ({exit_coords[0]}, {exit_coords[1]}) 설정")

    # 출구 주변 높이 낮추기
    for r_off in range(-1, 2):
        for c_off in range(-1, 2):
            if r_off == 0 and c_off == 0: continue
            nr, nc = exit_coords[0] + r_off, exit_coords[1] + c_off
            if 0 <= nr < dungeon.shape[0] and 0 <= nc < dungeon.shape[1] and dungeon[nr, nc] == 1:
                 height_map[nr, nc] = max(1, height_map[nr, nc] // 2)

    return entrance, exit_coords


def find_path_bfs(dungeon, height_map, feature_map, entrance, exit_coords, max_height_diff=4):
    """BFS 경로 탐색 및 경로 타일에 특성 부여"""
    if entrance is None or exit_coords is None: return None, None

    height, width = dungeon.shape
    visited = np.zeros_like(dungeon, dtype=bool)
    parent = {} # 경로 역추적용
    queue = deque([(entrance[0], entrance[1])])

    if not (0 <= entrance[0] < height and 0 <= entrance[1] < width and dungeon[entrance[0], entrance[1]] == 1):
        # 유효하지 않은 입구 처리 (가까운 유효 타일 찾기)
        # ... (map4.py와 동일, 생략) ...
        print(f"오류: 입구 {entrance}가 유효하지 않습니다.")
        # 간단히 가장 가까운 유효 타일 찾기
        q = deque([entrance])
        visited_start_find = {entrance}
        new_entrance = None
        while q:
            y, x = q.popleft()
            if dungeon[y,x] == 1:
                new_entrance = (y,x)
                break
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                 ny, nx = y+dy, x+dx
                 if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in visited_start_find:
                     q.append((ny,nx))
                     visited_start_find.add((ny,nx))
        
        if new_entrance:
            print(f"가장 가까운 유효 입구 {new_entrance}로 변경.")
            entrance = new_entrance
            queue = deque([entrance]) # 큐 초기화
        else:
             print("유효 입구를 찾을 수 없습니다.")
             return None, None # 경로 탐색 불가
             
    visited[entrance[0], entrance[1]] = True
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    path_found = False
    
    while queue:
        y, x = queue.popleft()
        
        if (y, x) == exit_coords:
            path_found = True
            break
        
        current_h = height_map[y, x] if height_map[y, x] > 0 else 1
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            
            if (0 <= ny < height and 0 <= nx < width and 
                dungeon[ny, nx] == 1 and not visited[ny, nx]):
                
                next_h = height_map[ny, nx] if height_map[ny, nx] > 0 else 1
                height_diff = abs(next_h - current_h)
                
                if height_diff <= max_height_diff:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
                    parent[(ny, nx)] = (y, x) # 부모 노드 기록
    
    if not path_found:
        # 경로 못 찾은 경우, 문제 지점 찾기 (map4.py와 유사)
        problematic_points = []
        visited_coords = np.argwhere(visited & (dungeon == 1))
        for y, x in visited_coords:
            current_h = height_map[y, x] if height_map[y, x] > 0 else 1
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width and 
                    dungeon[ny, nx] == 1 and not visited[ny, nx]):
                    next_h = height_map[ny, nx] if height_map[ny, nx] > 0 else 1
                    height_diff = abs(next_h - current_h)
                    if height_diff > max_height_diff:
                        problematic_points.append(((y, x), (ny, nx), height_diff))
        return None, problematic_points

    # 경로 역추적
    path = []
    curr = exit_coords
    while curr != entrance:
        path.append(curr)
        # ===== 경로 타일에 특성 부여 (입구/출구 제외) =====
        if feature_map[curr[0], curr[1]] == FEATURE_NONE:
             feature_map[curr[0], curr[1]] = FEATURE_PATH 
        curr = parent[curr]
    path.append(entrance)
    path.reverse()
    
    # 입구/출구 특성 다시 확인
    feature_map[entrance[0], entrance[1]] = FEATURE_ENTRANCE
    feature_map[exit_coords[0], exit_coords[1]] = FEATURE_EXIT

    print(f"BFS 경로 찾음 (길이: {len(path)})")
    return path, None # 경로 반환, 문제 지점 없음


def adjust_heights_for_path(dungeon, height_map, problematic_points, max_height_diff=4):
    """높이 차이가 큰 문제 지점 조정 (map4.py와 동일)"""
    print(f"높이 차이가 너무 큰 {len(problematic_points)}개의 지점 조정 중 (최대 허용 차이: {max_height_diff})...")
    adjusted_count = 0
    for (y1, x1), (y2, x2), height_diff in problematic_points:
        h1 = height_map[y1, x1] if height_map[y1, x1] > 0 else 1
        h2 = height_map[y2, x2] if height_map[y2, x2] > 0 else 1
        actual_diff = abs(h1-h2)
        if actual_diff > max_height_diff:
            adjustment = (actual_diff - max_height_diff + 1) // 2
            new_h1, new_h2 = (h1 + adjustment, h2 - adjustment) if h1 < h2 else (h1 - adjustment, h2 + adjustment)
            height_map[y1, x1] = max(1, new_h1)
            height_map[y2, x2] = max(1, new_h2)
            adjusted_count += 1
    print(f"{adjusted_count}개 지점의 높이 조정 완료.")
    return height_map


def ensure_path_exists(dungeon, height_map, feature_map, entrance, exit_coords, max_attempts=10, max_height_diff=4):
    """경로 보장 및 높이 조정 (map4.py와 유사, feature_map 전달)"""
    if entrance is None or exit_coords is None:
        print("입구 또는 출구 설정 불가.")
        return height_map, feature_map, None

    for attempt in range(max_attempts):
        print(f"경로 확인 시도 #{attempt+1}/{max_attempts}...")
        path, problematic_points = find_path_bfs(dungeon, height_map, feature_map, entrance, exit_coords, max_height_diff)
        
        if path:
            print(f"경로 발견!")
            return height_map, feature_map, path
        
        if problematic_points:
            print(f"경로 없음. {len(problematic_points)}개 지점 높이 조정 시도...")
            height_map = adjust_heights_for_path(dungeon, height_map, problematic_points, max_height_diff)
        else:
            print("경로 및 조정 지점 없음. 강제 조정 시도...")
            # 강제 조정 로직 (map4.py와 동일)
            if entrance: height_map[entrance[0], entrance[1]] = 1
            if exit_coords: height_map[exit_coords[0], exit_coords[1]] = 1
            for r_off in range(-1, 2):
                for c_off in range(-1, 2):
                    if entrance:
                        nr, nc = entrance[0] + r_off, entrance[1] + c_off
                        if 0 <= nr < dungeon.shape[0] and 0 <= nc < dungeon.shape[1] and dungeon[nr, nc] == 1: height_map[nr, nc] = 1
                    if exit_coords:
                        nr, nc = exit_coords[0] + r_off, exit_coords[1] + c_off
                        if 0 <= nr < dungeon.shape[0] and 0 <= nc < dungeon.shape[1] and dungeon[nr, nc] == 1: height_map[nr, nc] = 1

    # 마지막 시도
    path, _ = find_path_bfs(dungeon, height_map, feature_map, entrance, exit_coords, max_height_diff)
    if path:
        print("최종 조정 후 경로 발견!")
        return height_map, feature_map, path
    else:
        print(f"경고: {max_attempts}번 시도 후에도 경로 보장 실패.")
        return height_map, feature_map, None # 경로 없이 반환

# ================== 메인 실행 부분 ==================

# 던전 생성 파라미터 (map4와 유사)
width, height = 70, 60
room_count = 7
room_min, room_max = 7, 14
min_room_distance = 5
max_height = 18
max_height_diff = 4

# 던전 생성 시도
max_dungeon_attempts = 5
final_dungeon, final_rooms, final_path = None, None, None
final_height_map, final_feature_map = None, None
final_entrance, final_exit = None, None

for dungeon_attempt in range(max_dungeon_attempts):
    print(f"\n===== 던전 생성 시도 #{dungeon_attempt + 1} =====")
    dungeon, rooms, corridors = generate_dungeon(width, height, room_count, room_min, room_max, min_room_distance)

    if len(rooms) < 2: print("방 부족, 재시도."); continue

    height_map, feature_map = generate_height_map(dungeon, rooms, corridors, max_height=max_height)
    entrance, exit_coords = select_entrance_exit(dungeon, rooms, height_map, feature_map)

    if entrance is None or exit_coords is None: print("입/출구 생성 실패, 재시도."); continue

    print(f"입구: {entrance}, 출구: {exit_coords}")

    # 경로 확인 및 높이 조정 (feature_map 사용)
    adjusted_height_map, adjusted_feature_map, path = ensure_path_exists(
        dungeon, height_map.copy(), feature_map.copy(), entrance, exit_coords, max_height_diff=max_height_diff
    )

    if path:
        print("성공적인 던전 생성 완료!")
        final_dungeon, final_rooms = dungeon, rooms
        final_height_map, final_feature_map = adjusted_height_map, adjusted_feature_map
        final_entrance, final_exit, final_path = entrance, exit_coords, path
        break
    else:
        print("경로 보장 실패, 재시도.")

if final_path is None:
    print(f"오류: {max_dungeon_attempts}번 시도에도 유효 경로 던전 생성 실패.")
    exit()


# ===== 시각화 부분 =====
tile_width, tile_height = 1.5, 0.75
height_scale = 0.4

fig, ax = plt.subplots(figsize=(22, 18), dpi=150) # 그림 크기/해상도 증가
ax.set_aspect('equal')
ax.axis('off')

# 바닥면
floor_poly = Polygon([
    (-width * tile_width/2, height * tile_height/2), (0, 0),
    (width * tile_width/2, height * tile_height/2), (0, height * tile_height)
], closed=True, alpha=0.15, facecolor='#404040', edgecolor='gray')
ax.add_patch(floor_poly)

# 컬러맵
colors = ['#2c3e50', '#3498db', '#1abc9c', '#f1c40f', '#e67e22', '#e74c3c']
cmap = LinearSegmentedColormap.from_list('custom_terrain', colors, N=256)

# 타일 정보 수집 및 정렬
tiles_to_draw = []
for y in range(height):
    for x in range(width):
        if final_dungeon[y, x] == 1:
            h = final_height_map[y, x]
            feature = final_feature_map[y, x]
            iso_x = (x - y) * tile_width / 2
            iso_y = (x + y) * tile_height / 2
            tiles_to_draw.append({'x': x, 'y': y, 'h': h, 'feature': feature, 'iso_x': iso_x, 'iso_y': iso_y})

tiles_to_draw.sort(key=lambda t: (t['y'], t['x'], -t['h'])) # Y -> X -> -H 순서

# 타일 그리기 및 텍스트 추가
max_h_val = max(1, final_height_map.max())
feature_texts = [] # 텍스트 겹침 방지용

for tile in tiles_to_draw:
    x, y, h, feature = tile['x'], tile['y'], tile['h'], tile['feature']
    iso_x, iso_y = tile['iso_x'], tile['iso_y']
    
    normalized_height = h / max_h_val
    base_color = cmap(normalized_height)
    
    top_y_offset = iso_y - h * height_scale
    top_coord = (iso_x, top_y_offset)
    right_coord = (iso_x + tile_width/2, top_y_offset + tile_height/2)
    bottom_coord = (iso_x, top_y_offset + tile_height)
    left_coord = (iso_x - tile_width/2, top_y_offset + tile_height/2)
    
    # 옆면 그리기
    left_side_color = tuple(c*0.5 for c in base_color[:3]) + (base_color[3],)
    right_side_color = tuple(c*0.7 for c in base_color[:3]) + (base_color[3],)
    
    # ===== 다리 옆면 색상 변경 =====
    if feature == FEATURE_BRIDGE:
         left_side_color = '#8B4513' # 갈색 (나무 느낌)
         right_side_color = '#A0522D' # 약간 밝은 갈색
         
    if h > 0:
        left_side_coords = [left_coord, (left_coord[0], iso_y + tile_height/2), (iso_x, iso_y + tile_height), bottom_coord]
        left_poly = Polygon(left_side_coords, closed=True, facecolor=left_side_color, edgecolor='black', linewidth=0.3)
        ax.add_patch(left_poly)
        
        right_side_coords = [bottom_coord, (iso_x, iso_y + tile_height), (right_coord[0], iso_y + tile_height/2), right_coord]
        right_poly = Polygon(right_side_coords, closed=True, facecolor=right_side_color, edgecolor='black', linewidth=0.3)
        ax.add_patch(right_poly)

    # 윗면 그리기
    top_poly_coords = [top_coord, right_coord, bottom_coord, left_coord]
    
    tile_face_color = base_color
    tile_edge_color = 'black'
    tile_linewidth = 0.5
    text_label = None
    text_color = 'white'
    
    # ===== 특성별 색상 및 텍스트 설정 =====
    if feature == FEATURE_ENTRANCE:
        tile_face_color = 'lime'
        tile_linewidth = 1.5
        text_label = 'E' # 입구
        text_color = 'black'
    elif feature == FEATURE_EXIT:
        tile_face_color = 'red'
        tile_linewidth = 1.5
        border_poly = Polygon(top_poly_coords, closed=True, facecolor='none', edgecolor='yellow', linewidth=2.5)
        ax.add_patch(border_poly)
        text_label = 'X' # 출구
        text_color = 'black'
    elif feature == FEATURE_PATH:
        tile_face_color = 'yellow'
        tile_linewidth = 0.8
        # text_label = '.' # 경로는 점으로 표시 (선택적, 너무 많으면 지저분)
    elif feature == FEATURE_TRAP:
        # tile_face_color = 'purple' # 함정은 보라색 (선택적)
        tile_edge_color = 'red' # 함정 테두리 강조
        tile_linewidth = 1.0
        text_label = 'T' # 함정
        text_color = 'red'
    elif feature == FEATURE_SECRET_HINT:
        # tile_face_color = 'cyan' # 비밀 힌트는 청록색 (선택적)
        tile_edge_color = 'blue'
        tile_linewidth = 1.0
        text_label = '?' # 비밀
        text_color = 'blue'
    elif feature == FEATURE_OBSTACLE:
        # tile_face_color = base_color # 색은 그대로 두거나 약간 변경
        tile_edge_color = 'gray'
        tile_linewidth = 1.0
        text_label = 'O' # 장애물
        text_color = 'dimgray'
    elif feature == FEATURE_BRIDGE:
        tile_face_color = '#D2B48C' # 다리는 연한 갈색 (나무 판자 느낌)
        tile_edge_color = '#8B4513' # 어두운 갈색 테두리
        tile_linewidth = 0.7
        text_label = 'B' # 다리 (선택적)
        text_color = 'black'
        
    top_poly = Polygon(top_poly_coords, closed=True, facecolor=tile_face_color, edgecolor=tile_edge_color, linewidth=tile_linewidth)
    ax.add_patch(top_poly)

    # ===== 텍스트 레이블 추가 =====
    if text_label:
         # 텍스트 위치: 타일 중심 약간 위
         text_x = iso_x
         text_y = top_y_offset - tile_height * 0.1 # 살짝 위로
         
         # 간단한 겹침 방지: 해당 위치에 이미 텍스트가 있으면 그리지 않음
         can_draw_text = True
         for tx, ty, tl in feature_texts:
              if abs(text_x - tx) < tile_width * 0.4 and abs(text_y - ty) < tile_height * 0.4:
                   can_draw_text = False; break
         
         if can_draw_text:
              ax.text(text_x, text_y, text_label, 
                      ha='center', va='center', fontsize=8, color=text_color, fontweight='bold',
                      bbox=dict(boxstyle='circle,pad=0.1', fc='white', alpha=0.6, ec='none')) # 배경 추가
              feature_texts.append((text_x, text_y, text_label))


# 축 범위 및 제목
ax.autoscale_view()
ax.set_title('개선된 쿼터뷰 던전 맵 v5 (특성 표시)', fontsize=18, pad=25)

plt.tight_layout()
plt.savefig('dungeon_map_v5.png', dpi=150, bbox_inches='tight')
plt.show() 