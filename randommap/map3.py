import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import heapq  # 다익스트라 알고리즘을 위한 우선순위 큐
from collections import deque  # BFS를 위한 큐

# 폰트 설정 (한글 표시)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 던전과 높이 맵 생성 함수
def generate_dungeon(width, height, room_count=8, room_min=8, room_max=15, min_room_distance=3):
    dungeon = np.zeros((height, width), dtype=int)
    rooms = []
    attempts = 0
    max_attempts = 100
    
    # 방 생성 (방 사이 간격 확보)
    while len(rooms) < room_count and attempts < max_attempts:
        w = np.random.randint(room_min, room_max)
        h = np.random.randint(room_min, room_max)
        x = np.random.randint(1, width - w - 1)
        y = np.random.randint(1, height - h - 1)
        
        # 새 방이 기존 방과 충분한 거리를 유지하는지 확인
        too_close = False
        for rx, ry, rw, rh in rooms:
            # 두 방 사이 최소 거리 확인 (가로, 세로 모두 min_room_distance 이상 떨어져야 함)
            if not (x + w + min_room_distance <= rx or rx + rw + min_room_distance <= x or
                   y + h + min_room_distance <= ry or ry + rh + min_room_distance <= y):
                too_close = True
                break
        
        attempts += 1
        if too_close:
            continue
            
        # 방 추가
        dungeon[y:y+h, x:x+w] = 1
        rooms.append((x, y, w, h))
    
    # 방 연결 (복도 생성) - 넓은 복도 (2칸 너비)
    for i in range(1, len(rooms)):
        x1, y1, w1, h1 = rooms[i-1]
        x2, y2, w2, h2 = rooms[i]
        cx1, cy1 = x1 + w1//2, y1 + h1//2
        cx2, cy2 = x2 + w2//2, y2 + h2//2
        
        if np.random.rand() < 0.5:
            # 수직 후 수평 연결 (2칸 너비)
            min_y, max_y = min(cy1, cy2), max(cy1, cy2)
            for corridor_y in range(min_y, max_y+1):
                dungeon[corridor_y, cx1] = 1
                dungeon[corridor_y, cx1+1] = 1  # 두 칸 너비
            
            min_x, max_x = min(cx1, cx2), max(cx1, cx2)
            for corridor_x in range(min_x, max_x+1):
                dungeon[cy2, corridor_x] = 1
                dungeon[cy2+1, corridor_x] = 1  # 두 칸 너비
        else:
            # 수평 후 수직 연결 (2칸 너비)
            min_x, max_x = min(cx1, cx2), max(cx1, cx2)
            for corridor_x in range(min_x, max_x+1):
                dungeon[cy1, corridor_x] = 1
                dungeon[cy1+1, corridor_x] = 1  # 두 칸 너비
            
            min_y, max_y = min(cy1, cy2), max(cy1, cy2)
            for corridor_y in range(min_y, max_y+1):
                dungeon[corridor_y, cx2] = 1
                dungeon[corridor_y, cx2+1] = 1  # 두 칸 너비
    
    return dungeon, rooms

def generate_height_map(dungeon, rooms, smoothness=5, max_height=15):
    # 노이즈 기반 높이 맵 생성
    noise = np.random.rand(*dungeon.shape)
    
    # 노이즈 부드럽게 만들기
    for _ in range(smoothness):
        noise = (noise +
                np.roll(noise, 1, axis=0) + np.roll(noise, -1, axis=0) +
                np.roll(noise, 1, axis=1) + np.roll(noise, -1, axis=1)) / 5
    
    # 던전 영역에만 높이 적용
    height_map = (noise * max_height).astype(int) * (dungeon == 1)
    
    # 각 방마다 크게 다른 높이 할당 (1~15 사이의 값)
    room_heights = np.random.randint(1, max_height + 1, size=len(rooms))
    
    # 방들은 내부 높이는 더 일정하게, 방마다 높이 차이는 더 크게 만들기
    for i, (x, y, w, h) in enumerate(rooms):
        # 이 방의 기본 높이 - 각 방마다 크게 다르게
        base_height = room_heights[i]
        
        # 방 전체를 일정한 높이로 설정 (아주 작은 변화만 추가)
        for ry in range(y, y+h):
            for rx in range(x, x+w):
                if dungeon[ry, rx] == 1:
                    # 가장자리는 아주 작은 높이 변화만 추가
                    if rx == x or rx == x+w-1 or ry == y or ry == y+h-1:
                        height_map[ry, rx] = base_height + np.random.randint(-1, 2) // 2  # 더 작은 변화
                    else:
                        # 내부는 거의 동일한 높이
                        height_map[ry, rx] = base_height
    
    # 복도 높이 조정 - 복도를 좀 더 낮게 만들기
    for y in range(1, dungeon.shape[0]-1):
        for x in range(1, dungeon.shape[1]-1):
            # 던전 타일인데 주변 8방향 중 빈 공간이 있으면 복도로 간주
            if dungeon[y, x] == 1:
                neighbors = dungeon[y-1:y+2, x-1:x+2].flatten()
                if 0 in neighbors:  # 주변에 빈 공간이 있으면 복도일 가능성이 높음
                    # 방 내부가 아닌 복도로 판단되는 경우
                    is_in_room = False
                    for rx, ry, rw, rh in rooms:
                        if rx+1 <= x <= rx+rw-2 and ry+1 <= y <= ry+rh-2:  # 방 내부(테두리 제외)
                            is_in_room = True
                            break
                    
                    if not is_in_room:
                        # 복도는 낮은 높이로 설정 (1~3)
                        height_map[y, x] = np.random.randint(1, 4)
    
    # 음수 높이 제거
    height_map = np.maximum(height_map, 0)
    
    return height_map

def select_entrance_exit(dungeon, rooms):
    """입구와 출구 선택 (같은 위치일 수도 있음)"""
    # 방들 중에서 랜덤하게 입구와 출구가 있을 방 선택
    entrance_room_idx = np.random.randint(0, len(rooms))
    
    # 20% 확률로 입구와 출구가 같은 방
    if np.random.rand() < 0.2:
        exit_room_idx = entrance_room_idx
    else:
        # 다른 방들 중에서 출구 선택
        other_rooms = list(range(len(rooms)))
        other_rooms.remove(entrance_room_idx)
        exit_room_idx = np.random.choice(other_rooms)
    
    # 선택된 방들의 정보
    entrance_room = rooms[entrance_room_idx]
    exit_room = rooms[exit_room_idx]
    
    # 방 내부의 랜덤한 위치 선택
    ex, ey, ew, eh = entrance_room
    entrance_x = np.random.randint(ex+1, ex+ew-1)
    entrance_y = np.random.randint(ey+1, ey+eh-1)
    
    xx, xy, xw, xh = exit_room
    exit_x = np.random.randint(xx+1, xx+xw-1)
    exit_y = np.random.randint(xy+1, xy+xh-1)
    
    return (entrance_y, entrance_x), (exit_y, exit_x)

def find_path_bfs(dungeon, height_map, entrance, exit, max_height_diff=3):
    """
    BFS를 사용해 입구에서 출구까지의 경로 찾기
    max_height_diff: 이동 가능한 최대 높이 차이
    """
    height, width = dungeon.shape
    visited = np.zeros_like(dungeon, dtype=bool)
    queue = deque([(entrance[0], entrance[1], [])])  # (y, x, path)
    visited[entrance[0], entrance[1]] = True
    
    # 이동 방향 (상, 하, 좌, 우)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        y, x, path = queue.popleft()
        current_path = path + [(y, x)]
        
        # 출구에 도달한 경우
        if (y, x) == exit:
            return current_path, None  # 경로 반환, 문제 지점 없음
        
        # 인접한 타일로 이동
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            
            # 맵 범위 내이고 던전 타일이며 아직 방문하지 않은 경우
            if (0 <= ny < height and 0 <= nx < width and 
                dungeon[ny, nx] == 1 and not visited[ny, nx]):
                
                # 높이 차이 계산
                height_diff = abs(height_map[ny, nx] - height_map[y, x])
                
                # 높이 차이가 너무 크면 이동 불가
                if height_diff <= max_height_diff:
                    visited[ny, nx] = True
                    queue.append((ny, nx, current_path))
    
    # 경로를 찾지 못한 경우, 높이 차이가 큰 문제 지점 찾기
    problematic_points = []
    # 방문한 모든 지점에서 인접한 방문하지 않은 지점 중 높이 차이가 큰 지점 찾기
    for y in range(height):
        for x in range(width):
            if visited[y, x] and dungeon[y, x] == 1:
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width and 
                        dungeon[ny, nx] == 1 and not visited[ny, nx]):
                        
                        height_diff = abs(height_map[ny, nx] - height_map[y, x])
                        if height_diff > max_height_diff:
                            problematic_points.append(((y, x), (ny, nx), height_diff))
    
    return None, problematic_points  # 경로 없음, 문제 지점 반환

def adjust_heights_for_path(dungeon, height_map, problematic_points, max_height_diff=3):
    """높이 차이가 큰 문제 지점들의 높이 조정"""
    print(f"높이 차이가 너무 큰 {len(problematic_points)}개의 지점 조정 중...")
    
    for (y1, x1), (y2, x2), height_diff in problematic_points:
        # 두 타일의 현재 높이
        h1 = height_map[y1, x1]
        h2 = height_map[y2, x2]
        
        # 높이 차이가 max_height_diff보다 큰 경우 조정
        if height_diff > max_height_diff:
            # 낮은 쪽을 높이기, 높은 쪽을 낮추기 병행
            target_diff = max_height_diff
            if h1 < h2:
                # h1을 조금 높이고, h2를 조금 낮춤
                new_h1 = h1 + (height_diff - target_diff) // 2
                new_h2 = h2 - (height_diff - target_diff) // 2
            else:
                # h2를 조금 높이고, h1을 조금 낮춤
                new_h2 = h2 + (height_diff - target_diff) // 2
                new_h1 = h1 - (height_diff - target_diff) // 2
            
            height_map[y1, x1] = max(1, new_h1)  # 최소 높이는 1
            height_map[y2, x2] = max(1, new_h2)
            
            print(f"  지점 ({y1},{x1})와 ({y2},{x2}) 사이의 높이 차이 {height_diff}를 {abs(height_map[y1, x1] - height_map[y2, x2])}로 조정")
    
    return height_map

def ensure_path_exists(dungeon, height_map, entrance, exit, max_attempts=5, max_height_diff=3):
    """입구에서 출구까지 경로가 존재하도록 높이 조정"""
    for attempt in range(max_attempts):
        print(f"경로 확인 시도 #{attempt+1}...")
        path, problematic_points = find_path_bfs(dungeon, height_map, entrance, exit, max_height_diff)
        
        if path:
            print(f"입구에서 출구까지 경로를 찾았습니다! (길이: {len(path)})")
            return height_map, path
        
        if problematic_points:
            print(f"경로가 없습니다. 높이 조정 시도...")
            height_map = adjust_heights_for_path(dungeon, height_map, problematic_points, max_height_diff)
        else:
            print("경로를 찾을 수 없고 조정할 지점도 없습니다.")
            break
    
    # 마지막 시도
    path, _ = find_path_bfs(dungeon, height_map, entrance, exit, max_height_diff)
    if path:
        print("높이 조정 후 경로를 찾았습니다!")
        return height_map, path
    else:
        print("모든 시도 후에도 경로를 찾을 수 없습니다.")
        return height_map, None

# 랜덤 시드 제거 - 매번 다른 던전 생성
# np.random.seed(42)

# 던전과 높이 맵 생성
width, height = 60, 50  # 더 넓은 공간으로 확장
dungeon, rooms = generate_dungeon(width, height, room_count=6, min_room_distance=3)  # 방 수 감소, 간격 설정
height_map = generate_height_map(dungeon, rooms, max_height=12)

# 입구와 출구 선택
entrance, exit = select_entrance_exit(dungeon, rooms)
print(f"입구 위치: {entrance}")
print(f"출구 위치: {exit}")

# 경로 확인 및 필요시 높이 조정
height_map, path = ensure_path_exists(dungeon, height_map, entrance, exit)

# 아이소메트릭 투영 파라미터
tile_width, tile_height = 1.5, 0.75  # 타일 크기 증가
height_scale = 0.5  # 높이 스케일 증가

# 그림 그리기
fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
ax.set_aspect('equal')
ax.axis('off')

# ============================== 바닥면 색상 설정 =================================
# 맵 바닥 - 여기서 색상을 변경하면 맵 전체 바닥(던전 타일이 그려질 배경)의 색상이 변경됩니다.
# alpha: 투명도 (0.0: 완전 투명, 1.0: 완전 불투명)
# facecolor: 바닥면 색상 ('lightgray'는 연한 회색, RGB Hex 코드로 '#333333'은 어두운 회색)
# edgecolor: 바닥면 테두리 색상
floor_poly = Polygon([
    (-width * tile_width/2, height * tile_height/2),
    (0, 0),
    (width * tile_width/2, height * tile_height/2),
    (0, height * tile_height)
], closed=True, alpha=0.2, facecolor='lightgray', edgecolor='gray')
ax.add_patch(floor_poly)
# ==============================================================================

# 커스텀 컬러맵 생성 (더 선명한 대비)
# 타일의 높이에 따른 색상 그라데이션을 정의합니다.
# 첫 번째 색상(파란색 계열)이 낮은 높이, 마지막 색상(주황색 계열)이 높은 높이입니다.
colors = [(0.2, 0.5, 0.7), (0.3, 0.7, 0.3), (0.7, 0.8, 0.2), (0.8, 0.4, 0.2)]
cmap = LinearSegmentedColormap.from_list('custom_terrain', colors, N=256)

# 정렬 순서를 위해 좌표와 높이 리스트 만들기
tiles = []
for y in range(height):
    for x in range(width):
        if dungeon[y, x] == 1:
            h = height_map[y, x]
            iso_x = (x - y) * tile_width / 2
            iso_y = (x + y) * tile_height / 2
            tiles.append((x, y, h, iso_x, iso_y))

# 뒤에서 앞으로 그리기 위해 정렬 (먼 타일부터 그리기)
tiles.sort(key=lambda t: (t[0] + t[1], -t[2]))

# 경로 좌표들을 집합으로 변환 (빠른 검색을 위해)
path_set = set()
if path:
    path_set = set(path)

# 타일 그리기
for x, y, h, iso_x, iso_y in tiles:
    # 높이에 따른 색상
    normalized_height = h / max(1, height_map.max())
    color = cmap(normalized_height)  # 타일 상단 색상은 높이에 따라 자동으로 결정됩니다
    
    # 타일 윗면
    top_y = iso_y - h * height_scale  # 높이 적용
    
    top = (iso_x, top_y)
    right = (iso_x + tile_width/2, top_y + tile_height/2)
    bottom = (iso_x, top_y + tile_height)
    left = (iso_x - tile_width/2, top_y + tile_height/2)
    
    # ========================= 타일의 측면 색상 설정 ===========================
    # 측면 그리기 (항상 그림)
    # 왼쪽 면 - 타일의 윗면 색상에서 어둡게 변형한 색상 (원래 색상의 60%)
    left_side = [
        left,
        (left[0], iso_y + tile_height/2),  # 바닥 높이
        (iso_x, iso_y + tile_height),      # 바닥 높이
        bottom
    ]
    left_poly = Polygon(left_side, closed=True)
    ax.add_patch(left_poly)
    left_poly.set_facecolor(tuple(c*0.4 for c in color[:3]) + (color[3],))  # 어둡게
    left_poly.set_edgecolor('white')
    left_poly.set_linewidth(0.5)
    
    # 오른쪽 면 - 타일의 윗면 색상에서 약간 어둡게 변형한 색상 (원래 색상의 80%)
    right_side = [
        bottom,
        (iso_x, iso_y + tile_height),      # 바닥 높이
        (right[0], iso_y + tile_height/2), # 바닥 높이
        right
    ]
    right_poly = Polygon(right_side, closed=True)
    ax.add_patch(right_poly)
    right_poly.set_facecolor(tuple(c*0.6 for c in color[:3]) + (color[3],))  # 약간 어둡게
    right_poly.set_edgecolor('lightgray')
    right_poly.set_linewidth(0.5)
    # ==========================================================================
    
    # ========================= 타일의 윗면 색상 설정 ===========================
    # 윗면 그리기 (마지막에 그려서 겹치도록)
    # 타일의 윗면 색상은 높이에 따라 자동으로 결정됨 (낮음: 파란색 계열, 높음: 주황색 계열)
    
    # 경로, 입구, 출구 여부에 따라 타일 색상 조정
    if (y, x) == entrance:
        tile_color = 'green'  # 입구는 초록색
        tile_edge = 'black'
        linewidth = 1.5
    elif (y, x) == exit:
        tile_color = 'red'  # 출구는 빨간색
        tile_edge = 'black'
        linewidth = 1.5
    elif path and (y, x) in path_set:
        tile_color = 'yellow'  # 경로는 노란색
        tile_edge = 'black'
        linewidth = 0.7
    else:
        tile_color = color
        tile_edge = 'black'
        linewidth = 0.5
    
    top_poly = Polygon([top, right, bottom, left], closed=True)
    ax.add_patch(top_poly)
    top_poly.set_facecolor(tile_color)
    top_poly.set_edgecolor(tile_edge)
    top_poly.set_linewidth(linewidth)
    # ==========================================================================

# 축 범위 자동 설정
ax.autoscale_view()
        
# 제목 설정
ax.set_title('Quarter View Dungeon Map with Entrance/Exit', fontsize=16, pad=20)

plt.tight_layout()
plt.savefig('dungeon_map.png', dpi=120, bbox_inches='tight')
plt.show() 