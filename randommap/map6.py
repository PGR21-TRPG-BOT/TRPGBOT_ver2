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

# ===== 지형 특성 관련 상수 =====
FEATURE_NONE = 0
FEATURE_TRAP = 1          # 함정
FEATURE_SECRET_HINT = 2   # 비밀 통로 힌트
FEATURE_OBSTACLE = 3      # 장애물 (복도)
FEATURE_BRIDGE = 4        # 다리
FEATURE_ENTRANCE = 5      # 입구
FEATURE_EXIT = 6          # 출구
FEATURE_PATH = 7          # 주 경로
FEATURE_TREASURE = 8      # 보물

# ===========================

# 던전과 높이 맵 생성 함수
def generate_dungeon(width, height, room_count=8, room_min=8, room_max=15, min_room_distance=4, corridor_width_options=[1, 2], extra_connection_prob=0.3):
    # ... (map5.py와 동일) ...
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
    # ... (map5.py와 동일) ...
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

def generate_height_map(dungeon, rooms, corridors, smoothness=3, max_height=15, 
                        corridor_height_range=(1, 4), obstacle_prob=0.05, obstacle_height_range=(1, 3),
                        trap_prob = 0.02, secret_hint_prob = 0.1, treasure_prob = 0.015): # 확률 인자 추가
    height, width = dungeon.shape
    noise = np.random.rand(height, width)
    
    for _ in range(smoothness):
        noise = (noise + np.roll(noise, 1, axis=0) + np.roll(noise, -1, axis=0) +
                 np.roll(noise, 1, axis=1) + np.roll(noise, -1, axis=1)) / 5
    
    height_map = (noise * max_height).astype(int) * (dungeon == 1)
    feature_map = np.zeros_like(dungeon, dtype=int)
    
    room_heights = np.random.randint(1, max_height + 1, size=len(rooms))
    
    # 방 내부 높이 설정
    for i, (x, y, w, h) in enumerate(rooms):
        base_height = room_heights[i]
        for ry in range(y, y + h):
            for rx in range(x, x + w):
                if dungeon[ry, rx] == 1:
                    is_edge = rx == x or rx == x + w - 1 or ry == y or ry == y + h - 1
                    height_map[ry, rx] = max(1, base_height + (np.random.randint(-1, 2) if is_edge else np.random.randint(-1, 1)))

    # 복도 높이 및 장애물/다리 특성 설정
    all_corridor_points = set(itertools.chain(*corridors)) if corridors else set()
    for r in range(height):
        for c in range(width):
            if dungeon[r, c] == 1 and (r, c) in all_corridor_points:
                 is_in_room_interior = False
                 for rx, ry, rw, rh in rooms:
                     if rx < c < rx + rw -1 and ry < r < ry + rh -1: is_in_room_interior = True; break
                 
                 if not is_in_room_interior:
                    # 복도 높이 설정
                    if height_map[r, c] == 0:
                       height_map[r, c] = np.random.randint(corridor_height_range[0], corridor_height_range[1] + 1)
                    else:
                       height_map[r, c] = max(corridor_height_range[0], min(corridor_height_range[1], height_map[r, c]))

                    # 장애물 생성
                    if np.random.rand() < obstacle_prob:
                         is_surrounded = True
                         for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                             nr, nc = r + dr, c + dc
                             if not (0 <= nr < height and 0 <= nc < width and dungeon[nr, nc] == 1 and (nr, nc) in all_corridor_points):
                                 is_surrounded = False; break
                         if is_surrounded:
                            height_map[r, c] += np.random.randint(obstacle_height_range[0], obstacle_height_range[1] + 1)
                            feature_map[r, c] = FEATURE_OBSTACLE
                            # print(f"복도 장애물 추가: ({r}, {c})")
                            
                    # 다리 지형 감지 (장애물이 아닐 경우)
                    elif feature_map[r, c] == FEATURE_NONE: 
                        is_narrow = False
                        h_neighbors = (dungeon[r, c-1] == 1) + (dungeon[r, c+1] == 1) if 0 < c < width - 1 else 0
                        v_neighbors = (dungeon[r-1, c] == 1) + (dungeon[r+1, c] == 1) if 0 < r < height - 1 else 0
                        if h_neighbors + v_neighbors == 2 and h_neighbors != 1: # 정확히 2개의 이웃이 있고, 수평/수직 중 하나로만 연결됨
                            is_narrow = True
                            
                        if is_narrow:
                             below_empty = (r + 1 < height and dungeon[r+1, c] == 0)
                             below_far = (r + 1 < height and dungeon[r+1, c] == 1 and height_map[r, c] - height_map[r+1, c] > 5)
                             if below_empty or below_far:
                                 feature_map[r, c] = FEATURE_BRIDGE
                                 # print(f"다리 지형 감지: ({r}, {c})")

    # ===== 보물 배치 (방 내부) =====
    for i, (x, y, w, h) in enumerate(rooms):
        for ry in range(y + 1, y + h - 1): # 방 내부 순회 (가장자리 제외)
            for rx in range(x + 1, x + w - 1):
                if dungeon[ry, rx] == 1 and feature_map[ry, rx] == FEATURE_NONE: # 바닥이고 다른 특성 없을 때
                    if np.random.rand() < treasure_prob:
                        feature_map[ry, rx] = FEATURE_TREASURE
                        print(f"보물 추가: ({ry}, {rx})")

    # ===== 함정(Trap) 및 비밀(Secret Hint) 배치 =====
    all_dungeon_points = np.argwhere(dungeon == 1)
    for y, x in all_dungeon_points:
        if feature_map[y, x] != FEATURE_NONE: continue # 이미 특성 있으면 건너뜀
            
        live_neighbors = 0
        is_on_edge = False
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
             ny, nx = y + dy, x + dx
             if 0 <= ny < height and 0 <= nx < width and dungeon[ny, nx] == 1:
                 live_neighbors += 1
             else: # 벽에 인접한 경우
                 is_on_edge = True 
                 
        # 더 정확한 가장자리 확인 (대각선 포함)
        if not is_on_edge:
             for dy, dx in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                  ny, nx = y + dy, x + dx
                  if not (0 <= ny < height and 0 <= nx < width and dungeon[ny, nx] == 1):
                      is_on_edge = True; break

        # 비밀 힌트: 막다른 길
        if live_neighbors == 1:
             if np.random.rand() < secret_hint_prob:
                 feature_map[y, x] = FEATURE_SECRET_HINT
                 print(f"비밀 힌트 추가: ({y}, {x})")
        # 함정: 복도(이웃 2) 또는 가장자리(이웃 3 이상 & 벽 근처)
        elif live_neighbors >= 2 : 
             if is_on_edge or live_neighbors == 2: # 복도이거나 (방/복도)가장자리
                  if np.random.rand() < trap_prob:
                      feature_map[y, x] = FEATURE_TRAP
                      print(f"함정 추가: ({y}, {x})")

    # 높이 최소값 보정
    height_map = np.maximum(height_map, 1) * (dungeon == 1)
    return height_map, feature_map

def select_entrance_exit(dungeon, rooms, height_map, feature_map):
    # ... (map5.py와 동일) ...
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
        # Prevent entrance and exit being the same tile in the same room
        retry_exit = False
        if xw > 2: exit_x = np.random.randint(xx + 1, xx + xw - 1); retry_exit=True
        if xh > 2: exit_y = np.random.randint(xy + 1, xy + xh - 1); retry_exit=True
        if not retry_exit: # If room is 1xN or Nx1, we might need to pick another room
            if len(rooms) > 1:
                possible_exit_rooms = list(range(len(rooms)))
                possible_exit_rooms.remove(entrance_room_idx)
                exit_room_idx = np.random.choice(possible_exit_rooms)
                exit_room = rooms[exit_room_idx]
                xx, xy, xw, xh = exit_room
                exit_x = np.random.randint(xx + 1, xx + xw - 1) if xw > 2 else xx
                exit_y = np.random.randint(xy + 1, xy + xh - 1) if xh > 2 else xy
            else: # Only one room, unavoidable collision impossible if room > 1x1
                 break # Should be ok if room size >= 1x2 or 2x1
                
    entrance = (entrance_y, entrance_x)
    exit_coords = (exit_y, exit_x)

    # Ensure entrance/exit don't overwrite existing features if possible, or just set height/feature
    height_map[entrance[0], entrance[1]] = max(1, height_map[entrance[0], entrance[1]])
    height_map[exit_coords[0], exit_coords[1]] = 1
    
    # Overwrite feature map for E/X
    feature_map[entrance[0], entrance[1]] = FEATURE_ENTRANCE
    feature_map[exit_coords[0], exit_coords[1]] = FEATURE_EXIT
    
    print(f"입구 ({entrance[0]}, {entrance[1]}), 출구 ({exit_coords[0]}, {exit_coords[1]}) 설정")

    # Lower height around exit
    for r_off in range(-1, 2):
        for c_off in range(-1, 2):
            if r_off == 0 and c_off == 0: continue
            nr, nc = exit_coords[0] + r_off, exit_coords[1] + c_off
            if 0 <= nr < dungeon.shape[0] and 0 <= nc < dungeon.shape[1] and dungeon[nr, nc] == 1:
                 # Don't lower height of entrance if it's near exit
                 if (nr, nc) != entrance:
                     height_map[nr, nc] = max(1, height_map[nr, nc] // 2)

    return entrance, exit_coords

def find_path_bfs(dungeon, height_map, feature_map, entrance, exit_coords, max_height_diff=4):
    # ... (map5.py와 거의 동일, 경로 특성 부여 로직 유지) ...
    if entrance is None or exit_coords is None: return None, None

    height, width = dungeon.shape
    visited = np.zeros_like(dungeon, dtype=bool)
    parent = {} # 경로 역추적용
    queue = deque([(entrance[0], entrance[1])])

    if not (0 <= entrance[0] < height and 0 <= entrance[1] < width and dungeon[entrance[0], entrance[1]] == 1):
        print(f"오류: 입구 {entrance}가 유효하지 않습니다.")
        q = deque([entrance]); visited_start_find = {entrance}; new_entrance = None
        while q:
            y, x = q.popleft()
            if 0 <= y < height and 0 <= x < width and dungeon[y,x] == 1:
                new_entrance = (y,x); break
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                 ny, nx = y+dy, x+dx
                 if (ny, nx) not in visited_start_find:
                     q.append((ny,nx)); visited_start_find.add((ny,nx))
        if new_entrance: print(f"가장 가까운 유효 입구 {new_entrance}로 변경."); entrance = new_entrance; queue = deque([entrance])
        else: print("유효 입구를 찾을 수 없습니다."); return None, None
             
    visited[entrance[0], entrance[1]] = True
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    path_found = False
    
    while queue:
        y, x = queue.popleft()
        if (y, x) == exit_coords: path_found = True; break
        current_h = height_map[y, x] if height_map[y, x] > 0 else 1
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width and dungeon[ny, nx] == 1 and not visited[ny, nx]):
                next_h = height_map[ny, nx] if height_map[ny, nx] > 0 else 1
                if abs(next_h - current_h) <= max_height_diff:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
                    parent[(ny, nx)] = (y, x)
    
    if not path_found:
        problematic_points = []
        visited_coords = np.argwhere(visited & (dungeon == 1))
        for y, x in visited_coords:
            current_h = height_map[y, x] if height_map[y, x] > 0 else 1
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width and dungeon[ny, nx] == 1 and not visited[ny, nx]):
                    next_h = height_map[ny, nx] if height_map[ny, nx] > 0 else 1
                    if abs(next_h - current_h) > max_height_diff:
                        problematic_points.append(((y, x), (ny, nx), abs(next_h - current_h)))
        return None, problematic_points

    path = []; curr = exit_coords
    while curr != entrance:
        path.append(curr)
        # 경로 특성 부여 (기존 특성 덮어쓰지 않음 - 입구/출구/함정 등 유지)
        if feature_map[curr[0], curr[1]] == FEATURE_NONE:
             feature_map[curr[0], curr[1]] = FEATURE_PATH 
        curr = parent[curr]
    path.append(entrance); path.reverse()
    
    # 입구/출구 특성 재확인 (경로 탐색 중 FEATURE_PATH로 덮어쓰였을 수 있으므로)
    feature_map[entrance[0], entrance[1]] = FEATURE_ENTRANCE
    feature_map[exit_coords[0], exit_coords[1]] = FEATURE_EXIT

    print(f"BFS 경로 찾음 (길이: {len(path)})")
    return path, None

def adjust_heights_for_path(dungeon, height_map, problematic_points, max_height_diff=4):
    # ... (map5.py와 동일) ...
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
    # ... (map5.py와 동일) ...
    if entrance is None or exit_coords is None:
        print("입구 또는 출구 설정 불가.")
        return height_map, feature_map, None

    for attempt in range(max_attempts):
        print(f"경로 확인 시도 #{attempt+1}/{max_attempts}...")
        # 경로 찾을 때마다 feature_map이 업데이트 될 수 있으므로 copy 전달은 부적절할 수 있음
        # 원본 feature_map을 직접 수정하도록 변경
        path, problematic_points = find_path_bfs(dungeon, height_map, feature_map, entrance, exit_coords, max_height_diff)
        
        if path:
            print(f"경로 발견!")
            return height_map, feature_map, path
        
        if problematic_points:
            print(f"경로 없음. {len(problematic_points)}개 지점 높이 조정 시도...")
            height_map = adjust_heights_for_path(dungeon, height_map, problematic_points, max_height_diff)
        else:
            print("경로 및 조정 지점 없음. 강제 조정 시도...")
            if entrance: height_map[entrance[0], entrance[1]] = 1
            if exit_coords: height_map[exit_coords[0], exit_coords[1]] = 1
            for r_off in range(-1, 2): # 주변 1칸 강제 조정
                for c_off in range(-1, 2):
                    if entrance:
                        nr, nc = entrance[0] + r_off, entrance[1] + c_off
                        if 0 <= nr < dungeon.shape[0] and 0 <= nc < dungeon.shape[1] and dungeon[nr, nc] == 1: height_map[nr, nc] = 1
                    if exit_coords:
                        nr, nc = exit_coords[0] + r_off, exit_coords[1] + c_off
                        if 0 <= nr < dungeon.shape[0] and 0 <= nc < dungeon.shape[1] and dungeon[nr, nc] == 1: height_map[nr, nc] = 1

    path, _ = find_path_bfs(dungeon, height_map, feature_map, entrance, exit_coords, max_height_diff) # 마지막 시도
    if path: print("최종 조정 후 경로 발견!"); return height_map, feature_map, path
    else: print(f"경고: {max_attempts}번 시도 후에도 경로 보장 실패."); return height_map, feature_map, None

# ================== 메인 실행 부분 ==================

# 던전 생성 파라미터
width, height = 70, 60
room_count = 7
room_min, room_max = 8, 16 # 방 크기 약간 증가
min_room_distance = 5
max_height = 18
max_height_diff = 4
treasure_prob = 0.02 # 보물 등장 확률 증가

# 던전 생성 시도
max_dungeon_attempts = 5
final_dungeon, final_rooms, final_path = None, None, None
final_height_map, final_feature_map = None, None
final_entrance, final_exit = None, None

for dungeon_attempt in range(max_dungeon_attempts):
    print(f"\n===== 던전 생성 시도 #{dungeon_attempt + 1} =====")
    dungeon, rooms, corridors = generate_dungeon(width, height, room_count, room_min, room_max, min_room_distance, extra_connection_prob=0.35) # 곁가지 확률 증가

    if len(rooms) < 2: print("방 부족, 재시도."); continue

    height_map, feature_map = generate_height_map(dungeon, rooms, corridors, max_height=max_height, treasure_prob=treasure_prob)
    entrance, exit_coords = select_entrance_exit(dungeon, rooms, height_map, feature_map)

    if entrance is None or exit_coords is None: print("입/출구 생성 실패, 재시도."); continue

    print(f"입구: {entrance}, 출구: {exit_coords}")

    adjusted_height_map, adjusted_feature_map, path = ensure_path_exists(
        dungeon, height_map, feature_map, entrance, exit_coords, max_height_diff=max_height_diff
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

fig, ax = plt.subplots(figsize=(24, 20), dpi=150) # 그림 크기 증가
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

tiles_to_draw.sort(key=lambda t: (t['y'], t['x'], -t['h']))

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
    
    # 옆면 기본 색상
    left_side_color = tuple(c*0.5 for c in base_color[:3]) + (base_color[3],)
    right_side_color = tuple(c*0.7 for c in base_color[:3]) + (base_color[3],)
    
    # 윗면 기본 설정
    tile_face_color = base_color
    tile_edge_color = 'black'
    tile_linewidth = 0.5
    text_label = None
    text_color = 'white'
    
    # ===== 특성별 처리 =====
    if feature == FEATURE_BRIDGE:
         # 다리 옆면 색상 (나무/돌 느낌)
         left_side_color = '#A0522D' # 중간 갈색
         right_side_color = '#8B4513' # 어두운 갈색
         # 다리 윗면
         tile_face_color = '#D2B48C' # 연한 갈색 (나무 판자)
         tile_edge_color = '#8B4513' # 어두운 갈색 테두리
         tile_linewidth = 1.2 # 테두리 굵게
         text_label = 'B'
         text_color = 'black'
    elif feature == FEATURE_ENTRANCE:
        tile_face_color = 'lime'
        tile_linewidth = 1.5
        text_label = 'E'
        text_color = 'black'
    elif feature == FEATURE_EXIT:
        tile_face_color = 'red'
        tile_linewidth = 1.5
        text_label = 'X'
        text_color = 'black'
        # 출구 강조 테두리 (윗면 그린 후 추가)
    elif feature == FEATURE_PATH:
        tile_face_color = 'yellow'
        tile_linewidth = 0.8
    elif feature == FEATURE_TRAP:
        tile_edge_color = 'red'
        tile_linewidth = 1.0
        text_label = 'T'
        text_color = 'red'
    elif feature == FEATURE_SECRET_HINT:
        tile_edge_color = 'blue'
        tile_linewidth = 1.0
        text_label = '?'
        text_color = 'blue'
    elif feature == FEATURE_OBSTACLE:
        tile_edge_color = 'gray'
        tile_linewidth = 1.0
        text_label = 'O'
        text_color = 'dimgray'
    elif feature == FEATURE_TREASURE:
        tile_face_color = 'gold' # 금색 타일
        tile_edge_color = 'darkorange' # 주황색 테두리
        tile_linewidth = 1.0
        text_label = '$' # 보물 마커
        text_color = 'black'
        
    # 옆면 그리기 (높이가 있을 때)
    if h > 0:
        left_side_coords = [left_coord, (left_coord[0], iso_y + tile_height/2), (iso_x, iso_y + tile_height), bottom_coord]
        left_poly = Polygon(left_side_coords, closed=True, facecolor=left_side_color, edgecolor='black', linewidth=0.3)
        ax.add_patch(left_poly)
        
        right_side_coords = [bottom_coord, (iso_x, iso_y + tile_height), (right_coord[0], iso_y + tile_height/2), right_coord]
        right_poly = Polygon(right_side_coords, closed=True, facecolor=right_side_color, edgecolor='black', linewidth=0.3)
        ax.add_patch(right_poly)

    # 윗면 그리기
    top_poly_coords = [top_coord, right_coord, bottom_coord, left_coord]
    top_poly = Polygon(top_poly_coords, closed=True, facecolor=tile_face_color, edgecolor=tile_edge_color, linewidth=tile_linewidth)
    ax.add_patch(top_poly)

    # 출구 강조 테두리 (윗면 그린 후에)
    if feature == FEATURE_EXIT:
        border_poly = Polygon(top_poly_coords, closed=True, facecolor='none', edgecolor='yellow', linewidth=2.5)
        ax.add_patch(border_poly)
        
    # 텍스트 레이블 추가 (겹침 방지)
    if text_label:
         text_x = iso_x
         text_y = top_y_offset - tile_height * 0.1
         can_draw = True
         for tx, ty, _ in feature_texts:
              if abs(text_x - tx) < tile_width * 0.4 and abs(text_y - ty) < tile_height * 0.4:
                   can_draw = False; break
         if can_draw:
              ax.text(text_x, text_y, text_label, ha='center', va='center', 
                      fontsize=8, color=text_color, fontweight='bold',
                      bbox=dict(boxstyle='circle,pad=0.1', fc='white', alpha=0.6, ec='none'))
              feature_texts.append((text_x, text_y, text_label))

# 축 범위 및 제목
ax.autoscale_view()
ax.set_title('개선된 쿼터뷰 던전 맵 v6 (보물, 다리 개선)', fontsize=18, pad=25)

plt.tight_layout()
plt.savefig('dungeon_map_v6.png', dpi=150, bbox_inches='tight')
plt.show() 