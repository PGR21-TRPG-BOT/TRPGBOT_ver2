import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import heapq
from collections import deque
import itertools  # 방 연결 시 조합 사용

# 폰트 설정 (한글 표시)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 던전과 높이 맵 생성 함수
def generate_dungeon(width, height, room_count=8, room_min=8, room_max=15, min_room_distance=4, corridor_width_options=[1, 2]):
    dungeon = np.zeros((height, width), dtype=int)
    rooms = []
    attempts = 0
    max_attempts = 200 # 방 생성 시도 횟수 증가

    # 방 생성 (방 사이 간격 확보)
    while len(rooms) < room_count and attempts < max_attempts:
        w = np.random.randint(room_min, room_max)
        h = np.random.randint(room_min, room_max)
        x = np.random.randint(1, width - w - 1)
        y = np.random.randint(1, height - h - 1)

        new_room_rect = (x, y, w, h)
        
        # 새 방이 기존 방과 충분한 거리를 유지하는지 확인
        too_close = False
        for rx, ry, rw, rh in rooms:
            # 두 방 사이 최소 거리 확인 (확장된 영역 고려)
            if not (x + w + min_room_distance <= rx or rx + rw + min_room_distance <= x or
                   y + h + min_room_distance <= ry or ry + rh + min_room_distance <= y):
                too_close = True
                break
        
        attempts += 1
        if too_close:
            continue
            
        # 방 추가
        dungeon[y:y+h, x:x+w] = 1
        rooms.append(new_room_rect)

    if len(rooms) < 2:
        print("방을 충분히 생성하지 못했습니다.")
        return dungeon, rooms # 방이 2개 미만이면 복도 생성 불가

    # 방 연결 (모든 방 쌍을 고려하여 연결 시도) - MST(Minimum Spanning Tree) 유사 방식 사용
    connected = {0} # 연결된 방 인덱스 집합 (0번 방부터 시작)
    edges = [] # (거리, 방1 인덱스, 방2 인덱스)
    room_centers = [(r[0] + r[2]//2, r[1] + r[3]//2) for r in rooms]

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            dist = abs(room_centers[i][0] - room_centers[j][0]) + abs(room_centers[i][1] - room_centers[j][1])
            edges.append((dist, i, j))
    
    edges.sort() # 거리가 짧은 순서로 정렬
    
    corridors = [] # 생성된 복도 정보 저장 ((y, x) 좌표 리스트)

    num_edges = 0
    for dist, i, j in edges:
        if i in connected and j not in connected or i not in connected and j in connected:
            # 아직 연결되지 않은 두 컴포넌트를 연결하는 엣지 추가
            connected.add(i)
            connected.add(j)
            
            # 복도 생성
            x1, y1 = room_centers[i]
            x2, y2 = room_centers[j]
            
            corridor_width = np.random.choice(corridor_width_options) # 복도 너비 랜덤 선택
            
            points = [] # 현재 복도의 타일 좌표
            
            # 복도 생성 (L자 또는 Z자 형태 추가)
            if np.random.rand() < 0.7: # L자 복도 확률 증가
                # 수직 후 수평 또는 수평 후 수직 (L자)
                if np.random.rand() < 0.5:
                    # 수직 먼저
                    for cy in range(min(y1, y2), max(y1, y2) + 1):
                        for offset in range(corridor_width):
                            px = x1 + offset
                            if 0 <= px < width and 0 <= cy < height:
                                dungeon[cy, px] = 1
                                points.append((cy, px))
                    # 수평 나중
                    for cx in range(min(x1, x2), max(x1, x2) + 1):
                         for offset in range(corridor_width):
                            py = y2 + offset
                            if 0 <= py < height and 0 <= cx < width:
                                dungeon[py, cx] = 1
                                points.append((py, cx))
                else:
                    # 수평 먼저
                    for cx in range(min(x1, x2), max(x1, x2) + 1):
                        for offset in range(corridor_width):
                            py = y1 + offset
                            if 0 <= py < height and 0 <= cx < width:
                                dungeon[py, cx] = 1
                                points.append((py, cx))
                    # 수직 나중
                    for cy in range(min(y1, y2), max(y1, y2) + 1):
                         for offset in range(corridor_width):
                            px = x2 + offset
                            if 0 <= px < width and 0 <= cy < height:
                                dungeon[cy, px] = 1
                                points.append((cy, px))

            else: # Z자 복도 (중간 지점 추가)
                mid_x = np.random.randint(min(x1, x2), max(x1, x2) + 1) if x1 != x2 else x1
                mid_y = np.random.randint(min(y1, y2), max(y1, y2) + 1) if y1 != y2 else y1

                # y1 -> mid_y (수직)
                for cy in range(min(y1, mid_y), max(y1, mid_y) + 1):
                    for offset in range(corridor_width):
                        px = x1 + offset
                        if 0 <= px < width and 0 <= cy < height:
                            dungeon[cy, px] = 1
                            points.append((cy, px))
                # x1 -> mid_x (수평, mid_y 에서)
                for cx in range(min(x1, mid_x), max(x1, mid_x) + 1):
                    for offset in range(corridor_width):
                        py = mid_y + offset
                        if 0 <= py < height and 0 <= cx < width:
                            dungeon[py, cx] = 1
                            points.append((py, cx))
                # mid_y -> y2 (수직, mid_x 에서)
                for cy in range(min(mid_y, y2), max(mid_y, y2) + 1):
                    for offset in range(corridor_width):
                        px = mid_x + offset
                        if 0 <= px < width and 0 <= cy < height:
                            dungeon[cy, px] = 1
                            points.append((cy, px))
                # mid_x -> x2 (수평, y2 에서)
                for cx in range(min(mid_x, x2), max(mid_x, x2) + 1):
                     for offset in range(corridor_width):
                        py = y2 + offset
                        if 0 <= py < height and 0 <= cx < width:
                            dungeon[py, cx] = 1
                            points.append((py, cx))

            if points:
                corridors.append(points)
            num_edges += 1
            
            # 모든 방이 연결되면 종료 (최소 연결)
            if len(connected) == len(rooms):
                 # 추가 연결 (곁가지 생성 확률)
                 if np.random.rand() < 0.3 and len(edges) > num_edges: # 30% 확률로 곁가지 추가
                      dist, i, j = edges[num_edges] # 다음으로 짧은 엣지 사용
                      if i not in connected or j not in connected: # 아직 완전히 연결되지 않았다면 건너뜀
                          continue
                      
                      # 이미 연결된 컴포넌트 사이에 추가 복도 생성 (곁가지)
                      x1, y1 = room_centers[i]
                      x2, y2 = room_centers[j]
                      corridor_width = np.random.choice(corridor_width_options)
                      points = []
                      
                      # 간단한 L자 복도로 곁가지 생성
                      if np.random.rand() < 0.5:
                          for cy in range(min(y1, y2), max(y1, y2) + 1):
                              for offset in range(corridor_width):
                                  px = x1 + offset
                                  if 0 <= px < width and 0 <= cy < height: dungeon[cy, px] = 1; points.append((cy, px))
                          for cx in range(min(x1, x2), max(x1, x2) + 1):
                              for offset in range(corridor_width):
                                  py = y2 + offset
                                  if 0 <= py < height and 0 <= cx < width: dungeon[py, cx] = 1; points.append((py, cx))
                      else:
                          for cx in range(min(x1, x2), max(x1, x2) + 1):
                              for offset in range(corridor_width):
                                  py = y1 + offset
                                  if 0 <= py < height and 0 <= cx < width: dungeon[py, cx] = 1; points.append((py, cx))
                          for cy in range(min(y1, y2), max(y1, y2) + 1):
                              for offset in range(corridor_width):
                                  px = x2 + offset
                                  if 0 <= px < width and 0 <= cy < height: dungeon[cy, px] = 1; points.append((cy, px))
                                  
                      if points: corridors.append(points)
                      print(f"곁가지 복도 추가: 방 {i} <-> 방 {j}")
                      
                 # break # 최소 연결만 하려면 주석 해제

    return dungeon, rooms, corridors

def generate_height_map(dungeon, rooms, corridors, smoothness=3, max_height=15, corridor_height_range=(1, 4), obstacle_prob=0.05, obstacle_height_range=(1, 3)):
    height, width = dungeon.shape
    noise = np.random.rand(height, width)
    
    # 노이즈 부드럽게 만들기 (횟수 감소)
    for _ in range(smoothness):
        noise = (noise +
                 np.roll(noise, 1, axis=0) + np.roll(noise, -1, axis=0) +
                 np.roll(noise, 1, axis=1) + np.roll(noise, -1, axis=1)) / 5
    
    height_map = (noise * max_height).astype(int) * (dungeon == 1)
    
    # 각 방마다 다른 기본 높이 할당 (1~15)
    room_heights = np.random.randint(1, max_height + 1, size=len(rooms))
    
    # 방 내부 높이 설정 (가장자리는 약간 변화, 내부는 거의 일정)
    for i, (x, y, w, h) in enumerate(rooms):
        base_height = room_heights[i]
        for ry in range(y, y + h):
            for rx in range(x, x + w):
                if dungeon[ry, rx] == 1:
                    # 가장자리 약간의 변화, 안쪽은 거의 일정
                    if rx == x or rx == x + w - 1 or ry == y or ry == y + h - 1:
                        height_map[ry, rx] = max(1, base_height + np.random.randint(-1, 2))
                    else:
                         # 방 내부 높이 랜덤성 약간 추가
                        height_map[ry, rx] = max(1, base_height + np.random.randint(-1, 1)) # -1, 0 중 선택

    # 복도 높이 설정 (낮은 범위 내에서 랜덤)
    all_corridor_points = set(itertools.chain(*corridors)) if corridors else set()
    for r in range(height):
        for c in range(width):
            if dungeon[r, c] == 1 and (r, c) in all_corridor_points:
                 # 복도인지 확인 (방 내부는 제외)
                 is_in_room_interior = False
                 for rx, ry, rw, rh in rooms:
                     if rx < c < rx + rw -1 and ry < r < ry + rh -1:
                         is_in_room_interior = True
                         break
                 if not is_in_room_interior:
                    height_map[r, c] = np.random.randint(corridor_height_range[0], corridor_height_range[1] + 1)

                    # 복도 중간에 장애물(엄폐물) 추가
                    if np.random.rand() < obstacle_prob:
                         # 주변 타일이 복도인지 확인
                         is_surrounded_by_corridor = True
                         for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                             nr, nc = r + dr, c + dc
                             if not (0 <= nr < height and 0 <= nc < width and (nr, nc) in all_corridor_points):
                                 is_surrounded_by_corridor = False
                                 break
                         if is_surrounded_by_corridor: # 주변도 복도일 때만 장애물 생성
                            height_map[r, c] = height_map[r,c] + np.random.randint(obstacle_height_range[0], obstacle_height_range[1] + 1)
                            print(f"복도 장애물 추가: ({r}, {c}), 높이: {height_map[r, c]}")

    # 높이가 0인 타일 제거 (최소 높이 1)
    height_map = np.maximum(height_map, 1) * (dungeon == 1)
    
    return height_map

def select_entrance_exit(dungeon, rooms, height_map):
    """입구와 출구 선택, 출구 주변 높이 조정"""
    if not rooms: # 방이 없으면 입/출구 설정 불가
        return None, None
        
    # 방들 중에서 랜덤하게 입구와 출구가 있을 방 선택
    entrance_room_idx = np.random.randint(0, len(rooms))
    
    # 20% 확률로 입구와 출구가 같은 방, 또는 방이 하나뿐일 때
    if len(rooms) == 1 or np.random.rand() < 0.2:
        exit_room_idx = entrance_room_idx
    else:
        # 다른 방들 중에서 출구 선택
        possible_exit_rooms = list(range(len(rooms)))
        possible_exit_rooms.remove(entrance_room_idx)
        exit_room_idx = np.random.choice(possible_exit_rooms)
    
    # 선택된 방들의 정보
    entrance_room = rooms[entrance_room_idx]
    exit_room = rooms[exit_room_idx]
    
    # 방 내부의 랜덤한 위치 선택 (가장자리 제외)
    ex, ey, ew, eh = entrance_room
    entrance_x = np.random.randint(ex + 1, ex + ew - 1) if ew > 2 else ex
    entrance_y = np.random.randint(ey + 1, ey + eh - 1) if eh > 2 else ey
    
    xx, xy, xw, xh = exit_room
    exit_x = np.random.randint(xx + 1, xx + xw - 1) if xw > 2 else xx
    exit_y = np.random.randint(xy + 1, xy + xh - 1) if xh > 2 else xy

    # 입구와 출구가 같은 타일이 되지 않도록 보장
    while entrance_room_idx == exit_room_idx and entrance_x == exit_x and entrance_y == exit_y:
        if xw > 2 : exit_x = np.random.randint(xx + 1, xx + xw - 1)
        if xh > 2 : exit_y = np.random.randint(xy + 1, xy + xh - 1)
        
    entrance = (entrance_y, entrance_x)
    exit_coords = (exit_y, exit_x)

    # 출구 주변 타일 높이 특이하게 만들기 (예: 출구 타일만 높이 1)
    height_map[exit_y, exit_x] = 1 # 출구 타일은 낮게 설정
    print(f"출구 ({exit_y}, {exit_x}) 높이를 1로 설정")
    # 주변 3x3 영역 높이 약간 낮추기 (선택적)
    for r_off in range(-1, 2):
        for c_off in range(-1, 2):
            if r_off == 0 and c_off == 0: continue
            nr, nc = exit_y + r_off, exit_x + c_off
            if 0 <= nr < dungeon.shape[0] and 0 <= nc < dungeon.shape[1] and dungeon[nr, nc] == 1:
                 height_map[nr, nc] = max(1, height_map[nr, nc] // 2) # 주변 높이 절반으로 (최소 1)

    return entrance, exit_coords

def find_path_bfs(dungeon, height_map, entrance, exit_coords, max_height_diff=4): # 최대 높이 차이 증가
    """
    BFS를 사용해 입구에서 출구까지의 경로 찾기
    max_height_diff: 이동 가능한 최대 높이 차이 (증가됨)
    """
    if entrance is None or exit_coords is None: return None, None # 입/출구 없으면 경로 탐색 불가

    height, width = dungeon.shape
    visited = np.zeros_like(dungeon, dtype=bool)
    queue = deque([(entrance[0], entrance[1], [])])  # (y, x, path)
    
    # 입구가 던전 타일이 아니거나 높이맵 정보가 없는 경우 처리
    if not (0 <= entrance[0] < height and 0 <= entrance[1] < width and dungeon[entrance[0], entrance[1]] == 1):
        print(f"오류: 입구 {entrance}가 유효한 던전 타일이 아닙니다.")
        # 가장 가까운 유효한 던전 타일을 찾아 입구로 재설정 (간단한 방식)
        min_dist = float('inf')
        new_entrance = None
        for r in range(height):
            for c in range(width):
                if dungeon[r,c] == 1:
                    dist = abs(r - entrance[0]) + abs(c - entrance[1])
                    if dist < min_dist:
                        min_dist = dist
                        new_entrance = (r, c)
        if new_entrance:
             print(f"입구를 가장 가까운 유효 타일 {new_entrance}로 변경합니다.")
             entrance = new_entrance
             queue = deque([(entrance[0], entrance[1], [])])
        else:
             print("유효한 입구를 찾을 수 없습니다.")
             return None, None
             
    visited[entrance[0], entrance[1]] = True
    
    # 이동 방향 (상, 하, 좌, 우)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        y, x, path = queue.popleft()
        current_path = path + [(y, x)]
        
        # 출구에 도달한 경우
        if (y, x) == exit_coords:
            return current_path, None  # 경로 반환, 문제 지점 없음
        
        # 인접한 타일로 이동
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            
            # 맵 범위 내이고 던전 타일이며 아직 방문하지 않은 경우
            if (0 <= ny < height and 0 <= nx < width and 
                dungeon[ny, nx] == 1 and not visited[ny, nx]):
                
                # 높이 차이 계산 (현재 타일과 다음 타일 모두 높이 정보가 있는지 확인)
                current_h = height_map[y, x]
                next_h = height_map[ny, nx]
                
                # 가끔 높이맵에 0이 들어가는 경우 방지 (최소 높이 1 가정)
                if current_h == 0: current_h = 1
                if next_h == 0: next_h = 1
                
                height_diff = abs(next_h - current_h)
                
                # 높이 차이가 허용 범위 내면 이동
                if height_diff <= max_height_diff:
                    visited[ny, nx] = True
                    queue.append((ny, nx, current_path))
    
    # 경로를 찾지 못한 경우, 높이 차이가 큰 문제 지점 찾기
    problematic_points = []
    visited_coords = np.argwhere(visited & (dungeon == 1)) # 방문한 던전 타일 좌표

    for y, x in visited_coords:
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width and 
                dungeon[ny, nx] == 1 and not visited[ny, nx]):
                
                current_h = height_map[y, x] if height_map[y, x] > 0 else 1
                next_h = height_map[ny, nx] if height_map[ny, nx] > 0 else 1
                height_diff = abs(next_h - current_h)

                if height_diff > max_height_diff:
                    problematic_points.append(((y, x), (ny, nx), height_diff))
    
    return None, problematic_points  # 경로 없음, 문제 지점 반환

def adjust_heights_for_path(dungeon, height_map, problematic_points, max_height_diff=4): # 최대 높이 차이 증가
    """높이 차이가 큰 문제 지점들의 높이 조정 (조정 폭 개선)"""
    print(f"높이 차이가 너무 큰 {len(problematic_points)}개의 지점 조정 중 (최대 허용 차이: {max_height_diff})...")
    adjusted_count = 0
    
    for (y1, x1), (y2, x2), height_diff in problematic_points:
        # 두 타일의 현재 높이 (0이면 1로 간주)
        h1 = height_map[y1, x1] if height_map[y1, x1] > 0 else 1
        h2 = height_map[y2, x2] if height_map[y2, x2] > 0 else 1
        
        actual_diff = abs(h1-h2) # 실제 높이차 다시 계산

        # 높이 차이가 max_height_diff보다 큰 경우 조정
        if actual_diff > max_height_diff:
            # 조정량 계산 (차이의 절반 정도를 조정)
            adjustment = (actual_diff - max_height_diff + 1) // 2 # +1 하여 홀수 차이도 처리
            
            if h1 < h2:
                # h1을 높이고, h2를 낮춤
                new_h1 = h1 + adjustment
                new_h2 = h2 - adjustment
            else:
                # h2를 높이고, h1을 낮춤
                new_h2 = h2 + adjustment
                new_h1 = h1 - adjustment
            
            # 높이는 최소 1 이상이어야 함
            height_map[y1, x1] = max(1, new_h1)
            height_map[y2, x2] = max(1, new_h2)
            adjusted_count += 1
            
            # print(f"  지점 ({y1},{x1})[{h1}->{height_map[y1, x1]}] <-> ({y2},{x2})[{h2}->{height_map[y2, x2]}] 조정 (원래 차이: {actual_diff})")
    
    print(f"{adjusted_count}개 지점의 높이 조정 완료.")
    return height_map

def ensure_path_exists(dungeon, height_map, entrance, exit_coords, max_attempts=10, max_height_diff=4): # 시도 횟수 증가
    """입구에서 출구까지 경로가 존재하도록 높이 조정"""
    if entrance is None or exit_coords is None:
        print("입구 또는 출구가 설정되지 않아 경로를 보장할 수 없습니다.")
        return height_map, None

    for attempt in range(max_attempts):
        print(f"경로 확인 시도 #{attempt+1}/{max_attempts}...")
        path, problematic_points = find_path_bfs(dungeon, height_map, entrance, exit_coords, max_height_diff)
        
        if path:
            print(f"경로 발견! (길이: {len(path)})")
            return height_map, path
        
        if problematic_points:
            print(f"경로 없음. {len(problematic_points)}개 지점 높이 조정 시도...")
            height_map = adjust_heights_for_path(dungeon, height_map, problematic_points, max_height_diff)
        else:
            # 경로도 없고 문제 지점도 없는 경우 (예: 입구나 출구가 고립된 경우)
            print("경로를 찾을 수 없고 조정할 지점도 없습니다. 던전 구조 문제일 수 있습니다.")
            # 이 경우, 강제로 입구/출구 주변 높이를 1로 만들어 연결 시도 (최후의 수단)
            print("강제로 입구/출구 주변 높이를 1로 조정하여 재시도합니다...")
            if entrance: height_map[entrance[0], entrance[1]] = 1
            if exit_coords: height_map[exit_coords[0], exit_coords[1]] = 1
            # 주변 높이도 조정
            for r_off in range(-1, 2):
                for c_off in range(-1, 2):
                    if entrance:
                        nr, nc = entrance[0] + r_off, entrance[1] + c_off
                        if 0 <= nr < dungeon.shape[0] and 0 <= nc < dungeon.shape[1] and dungeon[nr, nc] == 1:
                             height_map[nr, nc] = 1
                    if exit_coords:
                        nr, nc = exit_coords[0] + r_off, exit_coords[1] + c_off
                        if 0 <= nr < dungeon.shape[0] and 0 <= nc < dungeon.shape[1] and dungeon[nr, nc] == 1:
                             height_map[nr, nc] = 1


    # 마지막 시도
    path, _ = find_path_bfs(dungeon, height_map, entrance, exit_coords, max_height_diff)
    if path:
        print("최종 높이 조정 후 경로를 찾았습니다!")
        return height_map, path
    else:
        print(f"경고: {max_attempts}번의 시도 후에도 경로를 찾을 수 없습니다. 던전 생성에 실패했을 수 있습니다.")
        return height_map, None

# ================== 메인 실행 부분 ==================

# 던전 생성 파라미터
width, height = 70, 60 # 맵 크기 증가
room_count = 7       # 방 개수 조정
room_min, room_max = 7, 14 # 방 크기 범위
min_room_distance = 5 # 방 사이 최소 거리 증가
max_height = 18      # 최대 높이 증가
max_height_diff = 4 # 이동 가능한 최대 높이 차이

# 던전 생성 시도 (유효한 던전이 나올 때까지)
max_dungeon_attempts = 5
final_dungeon, final_rooms, final_path = None, None, None

for dungeon_attempt in range(max_dungeon_attempts):
    print(f"\n===== 던전 생성 시도 #{dungeon_attempt + 1} =====")
    dungeon, rooms, corridors = generate_dungeon(width, height, room_count, room_min, room_max, min_room_distance)

    if len(rooms) < 2:
        print("방이 충분히 생성되지 않아 재시도합니다.")
        continue

    height_map = generate_height_map(dungeon, rooms, corridors, max_height=max_height)
    entrance, exit_coords = select_entrance_exit(dungeon, rooms, height_map)

    if entrance is None or exit_coords is None:
        print("입구 또는 출구를 생성하지 못해 재시도합니다.")
        continue

    print(f"입구: {entrance}, 출구: {exit_coords}")

    # 경로 확인 및 높이 조정
    adjusted_height_map, path = ensure_path_exists(dungeon, height_map.copy(), entrance, exit_coords, max_height_diff=max_height_diff)

    if path:
        print("성공적인 던전 생성 완료!")
        final_dungeon, final_rooms, final_height_map, final_entrance, final_exit, final_path = \
            dungeon, rooms, adjusted_height_map, entrance, exit_coords, path
        break # 성공 시 루프 종료
    else:
        print("이번 시도에서 경로를 보장하지 못했습니다. 재시도합니다.")

if final_path is None:
    print(f"오류: {max_dungeon_attempts}번의 시도에도 불구하고 유효한 경로를 가진 던전을 생성하지 못했습니다.")
    # 프로그램 종료 또는 기본 맵 사용 등의 처리
    exit()


# ===== 시각화 부분 =====
tile_width, tile_height = 1.5, 0.75
height_scale = 0.4 # 높이 스케일 조정

fig, ax = plt.subplots(figsize=(20, 16), dpi=120) # 그림 크기 및 해상도 증가
ax.set_aspect('equal')
ax.axis('off')

# 바닥면 그리기
floor_poly = Polygon([
    (-width * tile_width/2, height * tile_height/2), (0, 0),
    (width * tile_width/2, height * tile_height/2), (0, height * tile_height)
], closed=True, alpha=0.15, facecolor='#404040', edgecolor='gray') # 어두운 회색 바닥
ax.add_patch(floor_poly)

# 커스텀 컬러맵 (더 어둡고 다채로운 색상)
colors = ['#2c3e50', '#3498db', '#1abc9c', '#f1c40f', '#e67e22', '#e74c3c'] # 어두운 파랑 ~ 밝은 파랑 ~ 청록 ~ 노랑 ~ 주황 ~ 빨강
cmap = LinearSegmentedColormap.from_list('custom_terrain', colors, N=256)

# 타일 정보 수집
tiles_to_draw = []
for y in range(height):
    for x in range(width):
        if final_dungeon[y, x] == 1:
            h = final_height_map[y, x]
            iso_x = (x - y) * tile_width / 2
            iso_y = (x + y) * tile_height / 2
            tiles_to_draw.append({'x': x, 'y': y, 'h': h, 'iso_x': iso_x, 'iso_y': iso_y})

# 그리기 순서 정렬 (y좌표 -> x좌표 -> 높이 역순) - 더 정확한 가림 처리
tiles_to_draw.sort(key=lambda t: (t['y'], t['x'], -t['h']))

# 경로 좌표 집합
path_set = set(final_path) if final_path else set()

# 타일 그리기
max_h_val = max(1, final_height_map.max()) # 0으로 나누기 방지

for tile in tiles_to_draw:
    x, y, h, iso_x, iso_y = tile['x'], tile['y'], tile['h'], tile['iso_x'], tile['iso_y']
    
    normalized_height = h / max_h_val
    base_color = cmap(normalized_height)
    
    # 타일 윗면 좌표
    top_y_offset = iso_y - h * height_scale
    top_coord = (iso_x, top_y_offset)
    right_coord = (iso_x + tile_width/2, top_y_offset + tile_height/2)
    bottom_coord = (iso_x, top_y_offset + tile_height)
    left_coord = (iso_x - tile_width/2, top_y_offset + tile_height/2)
    
    # 타일 옆면 그리기 (높이가 0보다 클 때만)
    if h > 0:
        # 왼쪽 면 (더 어둡게)
        left_side_coords = [left_coord, (left_coord[0], iso_y + tile_height/2), (iso_x, iso_y + tile_height), bottom_coord]
        left_poly = Polygon(left_side_coords, closed=True, facecolor=tuple(c*0.5 for c in base_color[:3]) + (base_color[3],), edgecolor='black', linewidth=0.3)
        ax.add_patch(left_poly)
        
        # 오른쪽 면 (약간 어둡게)
        right_side_coords = [bottom_coord, (iso_x, iso_y + tile_height), (right_coord[0], iso_y + tile_height/2), right_coord]
        right_poly = Polygon(right_side_coords, closed=True, facecolor=tuple(c*0.7 for c in base_color[:3]) + (base_color[3],), edgecolor='black', linewidth=0.3)
        ax.add_patch(right_poly)

    # 타일 윗면 그리기
    top_poly_coords = [top_coord, right_coord, bottom_coord, left_coord]
    
    # 타일 색상 및 테두리 결정
    tile_face_color = base_color
    tile_edge_color = 'black'
    tile_linewidth = 0.5
    
    is_entrance = (y, x) == final_entrance
    is_exit = (y, x) == final_exit
    is_path = (y, x) in path_set
    
    if is_entrance:
        tile_face_color = 'lime' # 밝은 녹색
        tile_edge_color = 'black'
        tile_linewidth = 1.5
    elif is_exit:
        tile_face_color = 'red'
        tile_edge_color = 'black'
        tile_linewidth = 1.5
        # 출구 주변 강조 (노란색 테두리)
        border_poly = Polygon(top_poly_coords, closed=True, facecolor='none', edgecolor='yellow', linewidth=2.5)
        ax.add_patch(border_poly)
    elif is_path:
        tile_face_color = 'yellow'
        tile_edge_color = 'black'
        tile_linewidth = 0.8
    
    top_poly = Polygon(top_poly_coords, closed=True, facecolor=tile_face_color, edgecolor=tile_edge_color, linewidth=tile_linewidth)
    ax.add_patch(top_poly)

# 축 범위 자동 설정 및 여백 조정
ax.autoscale_view()
ax.set_title('개선된 쿼터뷰 던전 맵 (Improved Quarter View Dungeon Map)', fontsize=18, pad=25)

plt.tight_layout()
plt.savefig('dungeon_map_v4.png', dpi=150, bbox_inches='tight') # 고해상도 저장
plt.show() 