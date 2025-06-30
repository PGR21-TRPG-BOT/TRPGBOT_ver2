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
    # ... (map6.py와 동일) ...
    dungeon = np.zeros((height, width), dtype=int)
    rooms = []
    attempts = 0
    max_attempts = 200
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
                too_close = True; break
        attempts += 1
        if too_close: continue
        dungeon[y:y+h, x:x+w] = 1
        rooms.append(new_room_rect)
    if len(rooms) < 2: print("방 부족"); return dungeon, rooms, []
    connected = {0}; edges = []; room_centers = [(r[0] + r[2]//2, r[1] + r[3]//2) for r in rooms]
    for i in range(len(rooms)): 
        for j in range(i + 1, len(rooms)): 
            dist = abs(room_centers[i][0] - room_centers[j][0]) + abs(room_centers[i][1] - room_centers[j][1])
            edges.append((dist, i, j))
    edges.sort(); corridors = []; mst_edges_count = 0
    for dist, i, j in edges:
        if len(connected) == len(rooms): break
        if i in connected and j not in connected or i not in connected and j in connected:
            connected.add(i); connected.add(j)
            x1, y1 = room_centers[i]; x2, y2 = room_centers[j]
            corridor_width = np.random.choice(corridor_width_options)
            points = create_corridor(dungeon, x1, y1, x2, y2, corridor_width)
            if points: corridors.append(points)
            mst_edges_count += 1
    remaining_edges = edges[mst_edges_count:]; np.random.shuffle(remaining_edges)
    num_extra_connections = int(len(remaining_edges) * extra_connection_prob)
    for k in range(min(num_extra_connections, len(remaining_edges))):
        dist, i, j = remaining_edges[k]
        x1, y1 = room_centers[i]; x2, y2 = room_centers[j]
        print(f"곁가지 복도 시도: 방 {i} <-> 방 {j}")
        corridor_width = np.random.choice(corridor_width_options)
        points = create_corridor(dungeon, x1, y1, x2, y2, corridor_width)
        if points: corridors.append(points)
    return dungeon, rooms, corridors

def create_corridor(dungeon, x1, y1, x2, y2, width):
    # ... (map6.py와 동일) ...
    height, map_width = dungeon.shape; points = []
    if np.random.rand() < 0.7:
        if np.random.rand() < 0.5:
            for cy in range(min(y1, y2), max(y1, y2) + 1): 
                for offset in range(width): px=x1+offset; 
                if 0<=px<map_width and 0<=cy<height: dungeon[cy, px]=1; points.append((cy, px))
            for cx in range(min(x1, x2), max(x1, x2) + 1): 
                for offset in range(width): py=y2+offset; 
                if 0<=py<height and 0<=cx<map_width: dungeon[py, cx]=1; points.append((py, cx))
        else:
            for cx in range(min(x1, x2), max(x1, x2) + 1): 
                for offset in range(width): py=y1+offset; 
                if 0<=py<height and 0<=cx<map_width: dungeon[py, cx]=1; points.append((py, cx))
            for cy in range(min(y1, y2), max(y1, y2) + 1): 
                for offset in range(width): px=x2+offset; 
                if 0<=px<map_width and 0<=cy<height: dungeon[cy, px]=1; points.append((cy, px))
    else:
        mid_x = np.random.randint(min(x1,x2), max(x1,x2)+1) if x1!=x2 else x1
        mid_y = np.random.randint(min(y1,y2), max(y1,y2)+1) if y1!=y2 else y1
        for cy in range(min(y1, mid_y), max(y1, mid_y)+1): 
            for offset in range(width): px=x1+offset; 
            if 0<=px<map_width and 0<=cy<height: dungeon[cy, px]=1; points.append((cy, px))
        for cx in range(min(x1, mid_x), max(x1, mid_x)+1): 
            for offset in range(width): py=mid_y+offset; 
            if 0<=py<height and 0<=cx<map_width: dungeon[py, cx]=1; points.append((py, cx))
        for cy in range(min(mid_y, y2), max(mid_y, y2)+1): 
            for offset in range(width): px=mid_x+offset; 
            if 0<=px<map_width and 0<=cy<height: dungeon[cy, px]=1; points.append((cy, px))
        for cx in range(min(mid_x, x2), max(mid_x, x2)+1): 
            for offset in range(width): py=y2+offset; 
            if 0<=py<height and 0<=cx<map_width: dungeon[py, cx]=1; points.append((py, cx))
    return list(set(points))

def generate_height_map(dungeon, rooms, corridors, smoothness=3, max_height=15, 
                        corridor_height_range=(1, 4), obstacle_prob=0.05, obstacle_height_range=(1, 3),
                        trap_prob_base = 0.015, trap_prob_corridor_center=0.005, # 함정 확률 세분화
                        secret_hint_prob = 0.1, treasure_prob = 0.015):
    height, width = dungeon.shape
    noise = np.random.rand(height, width)
    for _ in range(smoothness): noise = (noise + np.roll(noise, 1, axis=0) + np.roll(noise, -1, axis=0) + np.roll(noise, 1, axis=1) + np.roll(noise, -1, axis=1)) / 5
    height_map = (noise * max_height).astype(int) * (dungeon == 1)
    feature_map = np.zeros_like(dungeon, dtype=int)
    room_heights = np.random.randint(1, max_height + 1, size=len(rooms))

    # 방 내부 높이
    for i, (x, y, w, h) in enumerate(rooms):
        base_height = room_heights[i]
        for ry in range(y, y + h): 
            for rx in range(x, x + w):
                if dungeon[ry, rx] == 1:
                    is_edge = rx == x or rx == x + w - 1 or ry == y or ry == y + h - 1
                    height_map[ry, rx] = max(1, base_height + (np.random.randint(-1, 2) if is_edge else np.random.randint(-1, 1)))

    # 복도 높이, 장애물, 다리
    all_corridor_points = set(itertools.chain(*corridors)) if corridors else set()
    for r in range(height): 
        for c in range(width):
            if dungeon[r, c] == 1 and (r, c) in all_corridor_points:
                 is_in_room_interior = any(rx < c < rx + rw - 1 and ry < r < ry + rh - 1 for rx, ry, rw, rh in rooms)
                 if not is_in_room_interior:
                    if height_map[r, c] == 0: height_map[r, c] = np.random.randint(corridor_height_range[0], corridor_height_range[1] + 1)
                    else: height_map[r, c] = max(corridor_height_range[0], min(corridor_height_range[1], height_map[r, c]))
                    if np.random.rand() < obstacle_prob:
                         is_surrounded = all(0 <= r+dr < height and 0 <= c+dc < width and dungeon[r+dr, c+dc] == 1 and (r+dr, c+dc) in all_corridor_points for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])
                         if is_surrounded: height_map[r, c] += np.random.randint(obstacle_height_range[0], obstacle_height_range[1] + 1); feature_map[r, c] = FEATURE_OBSTACLE
                    elif feature_map[r, c] == FEATURE_NONE:
                        h_neighbors = (dungeon[r, c-1]==1) + (dungeon[r, c+1]==1) if 0<c<width-1 else 0
                        v_neighbors = (dungeon[r-1, c]==1) + (dungeon[r+1, c]==1) if 0<r<height-1 else 0
                        is_narrow = (h_neighbors + v_neighbors == 2 and h_neighbors != 1)
                        if is_narrow:
                             below_empty = (r + 1 < height and dungeon[r+1, c] == 0)
                             below_far = (r + 1 < height and dungeon[r+1, c] == 1 and height_map[r, c] - height_map[r+1, c] > 5)
                             if below_empty or below_far: feature_map[r, c] = FEATURE_BRIDGE

    # 보물 배치 (방 내부)
    for i, (x, y, w, h) in enumerate(rooms):
        for ry in range(y + 1, y + h - 1): 
            for rx in range(x + 1, x + w - 1):
                if dungeon[ry, rx] == 1 and feature_map[ry, rx] == FEATURE_NONE:
                    if np.random.rand() < treasure_prob: feature_map[ry, rx] = FEATURE_TREASURE; # print(f"보물: ({ry}, {rx})")

    # 함정 및 비밀 힌트 배치
    all_dungeon_points = np.argwhere(dungeon == 1)
    for y, x in all_dungeon_points:
        if feature_map[y, x] != FEATURE_NONE: continue
        live_neighbors = 0; is_on_edge = False
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
             ny, nx = y + dy, x + dx
             if 0 <= ny < height and 0 <= nx < width and dungeon[ny, nx] == 1: live_neighbors += 1
             else: is_on_edge = True
        if not is_on_edge: # 대각선까지 확인해서 가장자리 아님을 확정
             if any(not (0 <= y+dy < height and 0 <= x+dx < width and dungeon[y+dy, x+dx] == 1) for dy, dx in [(-1,-1),(-1,1),(1,-1),(1,1)]): is_on_edge = True

        if live_neighbors == 1: # 막다른 길
             if np.random.rand() < secret_hint_prob: feature_map[y, x] = FEATURE_SECRET_HINT; # print(f"비밀: ({y}, {x})")
        elif live_neighbors >= 2: # 통로 또는 방
            # 함정 확률 결정: 복도 중간(이웃2, 가장자리X)은 낮은 확률, 그 외는 기본 확률
            current_trap_prob = trap_prob_corridor_center if live_neighbors == 2 and not is_on_edge else trap_prob_base
            if np.random.rand() < current_trap_prob:
                feature_map[y, x] = FEATURE_TRAP; # print(f"함정: ({y}, {x})")

    height_map = np.maximum(height_map, 1) * (dungeon == 1)
    return height_map, feature_map

def select_entrance_exit(dungeon, rooms, height_map, feature_map):
    # ... (map6.py와 거의 동일) ...
    if not rooms: return None, None
    entrance_room_idx = np.random.randint(0, len(rooms))
    if len(rooms) == 1 or np.random.rand() < 0.2: exit_room_idx = entrance_room_idx
    else: possible_exit_rooms = list(range(len(rooms))); possible_exit_rooms.remove(entrance_room_idx); exit_room_idx = np.random.choice(possible_exit_rooms)
    entrance_room = rooms[entrance_room_idx]; exit_room = rooms[exit_room_idx]
    ex, ey, ew, eh = entrance_room; entrance_x = np.random.randint(ex + 1, ex + ew - 1) if ew > 2 else ex; entrance_y = np.random.randint(ey + 1, ey + eh - 1) if eh > 2 else ey
    xx, xy, xw, xh = exit_room; exit_x = np.random.randint(xx + 1, xx + xw - 1) if xw > 2 else xx; exit_y = np.random.randint(xy + 1, xy + xh - 1) if xh > 2 else xy
    while entrance_room_idx == exit_room_idx and entrance_x == exit_x and entrance_y == exit_y:
        retry_exit = False
        if xw > 2: exit_x = np.random.randint(xx + 1, xx + xw - 1); retry_exit=True
        if xh > 2: exit_y = np.random.randint(xy + 1, xy + xh - 1); retry_exit=True
        if not retry_exit:
            if len(rooms) > 1:
                possible_exit_rooms = list(range(len(rooms))); possible_exit_rooms.remove(entrance_room_idx); exit_room_idx = np.random.choice(possible_exit_rooms); exit_room = rooms[exit_room_idx]; xx, xy, xw, xh = exit_room
                exit_x = np.random.randint(xx + 1, xx + xw - 1) if xw > 2 else xx; exit_y = np.random.randint(xy + 1, xy + xh - 1) if xh > 2 else xy
            else: break
    entrance = (entrance_y, entrance_x); exit_coords = (exit_y, exit_x)
    height_map[entrance[0], entrance[1]] = max(1, height_map[entrance[0], entrance[1]]); height_map[exit_coords[0], exit_coords[1]] = 1
    feature_map[entrance[0], entrance[1]] = FEATURE_ENTRANCE; feature_map[exit_coords[0], exit_coords[1]] = FEATURE_EXIT
    print(f"입구 ({entrance[0]}, {entrance[1]}), 출구 ({exit_coords[0]}, {exit_coords[1]}) 설정")
    for r_off in range(-1, 2): 
        for c_off in range(-1, 2):
            if r_off == 0 and c_off == 0: continue
            nr, nc = exit_coords[0] + r_off, exit_coords[1] + c_off
            if 0 <= nr < dungeon.shape[0] and 0 <= nc < dungeon.shape[1] and dungeon[nr, nc] == 1 and (nr, nc) != entrance: height_map[nr, nc] = max(1, height_map[nr, nc] // 2)
    return entrance, exit_coords

def find_path_bfs(dungeon, height_map, feature_map, entrance, exit_coords, max_height_diff=4):
    # ... (map6.py와 거의 동일) ...
    if entrance is None or exit_coords is None: return None, None
    height, width = dungeon.shape; visited = np.zeros_like(dungeon, dtype=bool); parent = {}; queue = deque([(entrance[0], entrance[1])])
    if not (0 <= entrance[0] < height and 0 <= entrance[1] < width and dungeon[entrance[0], entrance[1]] == 1):
        print(f"오류: 입구 {entrance} 유효X."); q=deque([entrance]); visited_s={entrance}; new_e=None
        while q:
            y, x = q.popleft()
            if 0<=y<height and 0<=x<width and dungeon[y,x]==1: new_e=(y,x); break
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                 ny, nx = y+dy, x+dx
                 if (ny,nx) not in visited_s:
                     q.append((ny,nx))
                     visited_s.add((ny,nx))
        if new_e: print(f"유효 입구 {new_e}로 변경."); entrance=new_e; queue=deque([entrance])
        else: print("유효 입구 못찾음."); return None, None
    visited[entrance[0], entrance[1]] = True
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    path_found = False
    
    while queue:
        y, x = queue.popleft()
        if (y, x) == exit_coords:
            path_found = True
            break
        current_h = height_map[y,x] if height_map[y,x]>0 else 1
        for dy, dx in directions:
            ny, nx = y+dy, x+dx
            if (0<=ny<height and 0<=nx<width and dungeon[ny,nx]==1 and not visited[ny,nx]):
                next_h = height_map[ny,nx] if height_map[ny,nx]>0 else 1
                if abs(next_h-current_h) <= max_height_diff:
                    visited[ny,nx] = True
                    queue.append((ny,nx))
                    parent[(ny,nx)]=(y,x)
    if not path_found:
        problematic_points = []
        visited_coords = np.argwhere(visited & (dungeon==1))
        for y, x in visited_coords:
            current_h = height_map[y,x] if height_map[y,x]>0 else 1
            for dy, dx in directions:
                ny, nx = y+dy, x+dx
                if (0<=ny<height and 0<=nx<width and dungeon[ny,nx]==1 and not visited[ny,nx]):
                    next_h = height_map[ny,nx] if height_map[ny,nx]>0 else 1
                    if abs(next_h-current_h) > max_height_diff: problematic_points.append(((y,x),(ny,nx),abs(next_h-current_h)))
        return None, problematic_points
    path = []; curr = exit_coords
    while curr != entrance:
        path.append(curr)
        if feature_map[curr[0], curr[1]] == FEATURE_NONE: feature_map[curr[0], curr[1]] = FEATURE_PATH
        curr = parent[curr]
    path.append(entrance); path.reverse()
    feature_map[entrance[0], entrance[1]] = FEATURE_ENTRANCE; feature_map[exit_coords[0], exit_coords[1]] = FEATURE_EXIT
    print(f"BFS 경로 찾음 (길이: {len(path)})")
    return path, None

def adjust_heights_for_path(dungeon, height_map, problematic_points, max_height_diff=4):
    # ... (map6.py와 동일) ...
    print(f"높이 조정 ({len(problematic_points)}개 지점)...") ; adjusted_count = 0
    for (y1, x1), (y2, x2), height_diff in problematic_points:
        h1 = height_map[y1,x1] if height_map[y1,x1]>0 else 1; h2 = height_map[y2,x2] if height_map[y2,x2]>0 else 1
        actual_diff = abs(h1-h2)
        if actual_diff > max_height_diff:
            adjustment = (actual_diff-max_height_diff+1)//2
            new_h1, new_h2 = (h1+adjustment,h2-adjustment) if h1<h2 else (h1-adjustment,h2+adjustment)
            height_map[y1,x1] = max(1,new_h1); height_map[y2,x2] = max(1,new_h2); adjusted_count+=1
    print(f"{adjusted_count}개 조정 완료."); return height_map

def ensure_path_exists(dungeon, height_map, feature_map, entrance, exit_coords, max_attempts=10, max_height_diff=4):
    # ... (map6.py와 거의 동일) ...
    if entrance is None or exit_coords is None: print("입/출구 설정 불가."); return height_map, feature_map, None
    for attempt in range(max_attempts):
        print(f"경로 확인 #{attempt+1}/{max_attempts}...")
        path, problematic_points = find_path_bfs(dungeon, height_map, feature_map, entrance, exit_coords, max_height_diff)
        if path: print("경로 발견!"); return height_map, feature_map, path
        if problematic_points: print(f"경로 없음. {len(problematic_points)}개 높이 조정..."); height_map = adjust_heights_for_path(dungeon, height_map, problematic_points, max_height_diff)
        else: 
            print("경로/조정 지점 없음. 강제 조정...")
            if entrance: height_map[entrance[0], entrance[1]] = 1
            if exit_coords: height_map[exit_coords[0], exit_coords[1]] = 1
            for r_off, c_off in itertools.product([-1,0,1],[-1,0,1]):
                 if r_off==0 and c_off==0: continue
                 if entrance:
                     nrE,ncE = entrance[0]+r_off,entrance[1]+c_off
                     if 0<=nrE<dungeon.shape[0] and 0<=ncE<dungeon.shape[1] and dungeon[nrE,ncE]==1:
                         height_map[nrE,ncE]=1
                 if exit_coords:
                     nrX,ncX = exit_coords[0]+r_off,exit_coords[1]+c_off
                     if 0<=nrX<dungeon.shape[0] and 0<=ncX<dungeon.shape[1] and dungeon[nrX,ncX]==1:
                         height_map[nrX,ncX]=1
    path, _ = find_path_bfs(dungeon, height_map, feature_map, entrance, exit_coords, max_height_diff)
    if path: print("최종 조정 후 경로 발견!"); return height_map, feature_map, path
    else: print(f"경고: {max_attempts}번 시도 후 경로 보장 실패."); return height_map, feature_map, None

# ================== 메인 실행 부분 ==================

# 던전 생성 파라미터
width, height = 70, 60
room_count = 7
room_min, room_max = 8, 16
min_room_distance = 5
max_height = 18
max_height_diff = 4
trap_prob_base = 0.015      # 기본 함정 확률 (가장자리 등)
trap_prob_corridor_center = 0.005 # 복도 중앙 함정 확률 (낮춤)
treasure_prob = 0.02

# 던전 생성 시도
max_dungeon_attempts = 5
final_dungeon, final_rooms, final_path = None, None, None
final_height_map, final_feature_map = None, None
final_entrance, final_exit = None, None

for dungeon_attempt in range(max_dungeon_attempts):
    print(f"\n===== 던전 생성 시도 #{dungeon_attempt + 1} =====")
    dungeon, rooms, corridors = generate_dungeon(width, height, room_count, room_min, room_max, min_room_distance, extra_connection_prob=0.35)
    if len(rooms) < 2: print("방 부족, 재시도."); continue
    height_map, feature_map = generate_height_map(dungeon, rooms, corridors, max_height=max_height, 
                                                trap_prob_base=trap_prob_base, 
                                                trap_prob_corridor_center=trap_prob_corridor_center, 
                                                treasure_prob=treasure_prob)
    entrance, exit_coords = select_entrance_exit(dungeon, rooms, height_map, feature_map)
    if entrance is None or exit_coords is None: print("입/출구 생성 실패, 재시도."); continue
    print(f"입구: {entrance}, 출구: {exit_coords}")
    adjusted_height_map, adjusted_feature_map, path = ensure_path_exists(
        dungeon, height_map, feature_map, entrance, exit_coords, max_height_diff=max_height_diff)
    if path:
        print("성공적인 던전 생성 완료!")
        final_dungeon, final_rooms = dungeon, rooms
        final_height_map, final_feature_map = adjusted_height_map, adjusted_feature_map
        final_entrance, final_exit, final_path = entrance, exit_coords, path
        break
    else: print("경로 보장 실패, 재시도.")

if final_path is None: print(f"오류: {max_dungeon_attempts}번 시도에도 유효 경로 던전 생성 실패."); exit()

# ===== 시각화 부분 =====
tile_width, tile_height = 1.5, 0.75
height_scale = 0.4

fig, ax = plt.subplots(figsize=(24, 20), dpi=150)
ax.set_aspect('equal'); ax.axis('off')

# 바닥면
floor_poly = Polygon([
    (-width*tile_width/2, height*tile_height/2), (0,0), (width*tile_width/2, height*tile_height/2), (0, height*tile_height)],
    closed=True, alpha=0.15, facecolor='#404040', edgecolor='gray')
ax.add_patch(floor_poly)

# 컬러맵
colors = ['#2c3e50', '#3498db', '#1abc9c', '#f1c40f', '#e67e22', '#e74c3c']
cmap = LinearSegmentedColormap.from_list('custom_terrain', colors, N=256)

# 타일 정보 수집 및 정렬
tiles_to_draw = []
for y in range(height): 
    for x in range(width):
        if final_dungeon[y, x] == 1:
            h = final_height_map[y, x]; feature = final_feature_map[y, x]
            iso_x = (x - y) * tile_width / 2; iso_y = (x + y) * tile_height / 2
            tiles_to_draw.append({'x':x, 'y':y, 'h':h, 'feature':feature, 'iso_x':iso_x, 'iso_y':iso_y})

# ===== 렌더링 순서 변경 =====
tiles_to_draw.sort(key=lambda t: (t['y'] + t['x'], t['h'])) # y+x 오름차순, 높이 오름차순

# 타일 그리기 및 텍스트 추가
max_h_val = max(1, final_height_map.max())
feature_texts = []

for tile in tiles_to_draw:
    x, y, h, feature = tile['x'], tile['y'], tile['h'], tile['feature']
    iso_x, iso_y = tile['iso_x'], tile['iso_y']
    normalized_height = h / max_h_val; base_color = cmap(normalized_height)
    top_y_offset = iso_y - h * height_scale
    top_coord = (iso_x, top_y_offset); right_coord = (iso_x + tile_width/2, top_y_offset + tile_height/2)
    bottom_coord = (iso_x, top_y_offset + tile_height); left_coord = (iso_x - tile_width/2, top_y_offset + tile_height/2)
    
    # 옆면/윗면 기본값
    left_side_color = tuple(c*0.5 for c in base_color[:3]) + (base_color[3],)
    right_side_color = tuple(c*0.7 for c in base_color[:3]) + (base_color[3],)
    tile_face_color = base_color; tile_edge_color = 'black'; tile_linewidth = 0.5
    text_label = None; text_color = 'white'; text_fontsize = 7 # 텍스트 크기 약간 줄임
    
    # 특성별 처리
    if feature == FEATURE_BRIDGE:
         left_side_color = '#A0522D'; right_side_color = '#8B4513'
         tile_face_color = '#D2B48C'; tile_edge_color = '#8B4513'; tile_linewidth = 1.2
         text_label = 'B'; text_color = 'black'
    elif feature == FEATURE_ENTRANCE:
        tile_face_color = 'lime'; tile_linewidth = 1.5; text_label = 'E'; text_color = 'black'
    elif feature == FEATURE_EXIT:
        tile_face_color = 'red'; tile_linewidth = 1.5; text_label = 'X'; text_color = 'black'
    elif feature == FEATURE_PATH: tile_face_color = 'yellow'; tile_linewidth = 0.8
    elif feature == FEATURE_TRAP:
        tile_edge_color = 'red'; tile_linewidth = 1.0; text_label = 'T'; text_color = 'red'
    elif feature == FEATURE_SECRET_HINT:
        tile_edge_color = 'blue'; tile_linewidth = 1.0; text_label = '?'; text_color = 'blue'
    elif feature == FEATURE_OBSTACLE:
        tile_edge_color = 'gray'; tile_linewidth = 1.0; text_label = 'O'; text_color = 'dimgray'
    elif feature == FEATURE_TREASURE:
        tile_face_color = 'gold'; tile_edge_color = 'darkorange'; tile_linewidth = 1.0
        text_label = '$'; text_color = 'black'
        
    # 옆면 그리기 (높이 > 0)
    if h > 0:
        left_side_coords = [left_coord, (left_coord[0], iso_y + tile_height/2), (iso_x, iso_y + tile_height), bottom_coord]
        left_poly = Polygon(left_side_coords, closed=True, facecolor=left_side_color, edgecolor='black', linewidth=0.2) # 옆면 선 얇게
        ax.add_patch(left_poly)
        right_side_coords = [bottom_coord, (iso_x, iso_y + tile_height), (right_coord[0], iso_y + tile_height/2), right_coord]
        right_poly = Polygon(right_side_coords, closed=True, facecolor=right_side_color, edgecolor='black', linewidth=0.2) # 옆면 선 얇게
        ax.add_patch(right_poly)

    # 윗면 그리기
    top_poly_coords = [top_coord, right_coord, bottom_coord, left_coord]
    top_poly = Polygon(top_poly_coords, closed=True, facecolor=tile_face_color, edgecolor=tile_edge_color, linewidth=tile_linewidth)
    ax.add_patch(top_poly)

    # 출구 강조 테두리
    if feature == FEATURE_EXIT:
        border_poly = Polygon(top_poly_coords, closed=True, facecolor='none', edgecolor='yellow', linewidth=2.5)
        ax.add_patch(border_poly)
        
    # 텍스트 레이블 (겹침 방지)
    if text_label:
         text_x = iso_x; text_y = top_y_offset - tile_height * 0.1
         can_draw = True
         for tx, ty, _ in feature_texts:
              # 겹침 거리 기준 약간 넓힘
              if abs(text_x - tx) < tile_width * 0.5 and abs(text_y - ty) < tile_height * 0.5:
                   can_draw = False; break
         if can_draw:
              ax.text(text_x, text_y, text_label, ha='center', va='center', 
                      fontsize=text_fontsize, color=text_color, fontweight='bold',
                      bbox=dict(boxstyle='circle,pad=0.1', fc='white', alpha=0.7, ec='none')) # 투명도 약간 조절
              feature_texts.append((text_x, text_y, text_label))

# 축 범위 및 제목
ax.autoscale_view()
ax.set_title('개선된 쿼터뷰 던전 맵 v7 (함정 빈도 조정, 렌더링 순서 개선)', fontsize=18, pad=25)

plt.tight_layout()
plt.savefig('dungeon_map_v7.png', dpi=150, bbox_inches='tight')
plt.show() 