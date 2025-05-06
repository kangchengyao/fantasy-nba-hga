import pandas as pd
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import time
fitness_cache = {}  # 全域快取字典
# 讀取球員數據
player_data = pd.read_csv("sportsref_final.csv")


# 確保數據類型正確，並將涉及計算的數據欄位轉為數字類型
numeric_columns = ['PTS', 'AST', 'STL', 'BLK', 'TOV', 'FGA', 'FG', 'FTA', 'FT', '3P', 'ORB', 'DRB', 'PF']
for col in numeric_columns:
    player_data[col] = pd.to_numeric(player_data[col], errors='coerce').fillna(0)

# 設定球隊位置需求
positions_required = ['PG', 'SG', 'G', 'G', 'SF', 'PF', 'F', 'F', 'C', 'C', 'Util', 'Util', 'BN', 'BN', 'BN', 'IL', 'IL+']
target_players = {
    "james harden", "shai gilgeous-alexander", "anthony edwards", "trae young",
    "cade cunningham","jayson tatum", "giannis antetokounmpo", "karl-anthony towns",
    "lebron james","nikola jokic" 
}
# 設定Fantasy計分系統
scoring_system = {
    'FGA': -0.9,
    'FG': 2,  # 命中次數
    'FTA': -1.5,
    'FT': 2,  # 罰球命中次數
    '3P': 1.75,  # 三分命中次數
    'PTS': 0.75,  # 得分
    'OREB': 1.5,  # 進攻籃板
    'DRB': 1,     # 防守籃板
    'AST': 1.5,   # 助攻
    'STL': 3,     # 抄截
    'BLK': 3.5,   # 阻攻
    'TOV': -1,    # 失誤
    'PF': -0.25   # 個人犯規
}
# 計算某項數據達到 10 的概率
def calculate_probability(mean, std_dev):
    # 使用正態分布計算 P(Stat ≥ 10)
    if std_dev == 0:  # 防止標準差為 0
        return 1.0 if mean >= 10 else 0.0
    return 1 - stats.norm.cdf(10, loc=mean, scale=std_dev)
# 推估雙十和大三元的平均次數
def estimate_double_triple(player_stats):
    total_games = 82  # 直接取球隊的總出賽場次
    mean_pts = player_stats['PTS']
    mean_reb = player_stats['ORB'] + player_stats['DRB']
    mean_ast = player_stats['AST']
    mean_stl = player_stats['STL']
    mean_blk = player_stats['BLK']

    # 假設標準差為平均值的 10%（根據真實數據可以調整）
    std_dev_pts = mean_pts * 0.15
    std_dev_reb = mean_reb * 0.15
    std_dev_ast = mean_ast * 0.20
    std_dev_stl = mean_stl * 0.15
    std_dev_blk = mean_blk * 0.15

    # 計算每項數據達到 10 的概率
    p_pts = calculate_probability(mean_pts, std_dev_pts)
    p_reb = calculate_probability(mean_reb, std_dev_reb)
    p_ast = calculate_probability(mean_ast, std_dev_ast)
    p_stl = calculate_probability(mean_stl, std_dev_stl)
    p_blk = calculate_probability(mean_blk, std_dev_blk)

    # 雙十：兩項數據 ≥ 10 的概率（PTS + REB 或 REB + AST 等）
    double_double_probability = (
        p_pts * p_reb +
        p_pts * p_ast +
        p_reb * p_ast
    )

    # 大三元：三項數據 ≥ 10 的概率（PTS + REB + AST 等）
    triple_double_probability = (
        p_pts * p_reb * p_ast +
        p_reb * p_ast * p_blk +
        p_pts * p_ast * p_blk
    )

    # 返回平均次數（總次數除以場次）
    average_double_double = double_double_probability
    average_triple_double = triple_double_probability

    return average_double_double, average_triple_double

def fitness_function(team):
    total_score = 0
    position_penalty = 0
    selected_players = []
    seen_players = set()

    for i, player_id in enumerate(team):
        position = positions_required[i]

        if player_id in seen_players:
            continue

        # 檢查快取
        if player_id in fitness_cache:
            player_score, adjusted_score, injury_impact, no_injury_score, player_stats = fitness_cache[player_id]
        else:
            player_stats = player_data[player_data['Rk'] == player_id]
            if player_stats.empty:
                continue

            player_stats = player_stats.iloc[0]

            # 基本分數計算
            player_score = (
                player_stats['FGA'] * scoring_system['FGA'] +
                player_stats['FG'] * scoring_system['FG'] +
                player_stats['FTA'] * scoring_system['FTA'] +
                player_stats['FT'] * scoring_system['FT'] +
                player_stats['3P'] * scoring_system['3P'] +
                player_stats['PTS'] * scoring_system['PTS'] +
                player_stats['ORB'] * scoring_system['OREB'] +
                player_stats['DRB'] * scoring_system['DRB'] +
                player_stats['AST'] * scoring_system['AST'] +
                player_stats['STL'] * scoring_system['STL'] +
                player_stats['BLK'] * scoring_system['BLK'] +
                player_stats['TOV'] * scoring_system['TOV'] +
                player_stats['PF'] * scoring_system['PF']
            )

            # 雙十與大三元額外分數
            average_double_double, average_triple_double = estimate_double_triple(player_stats)
            player_score += average_double_double * 2.5
            player_score += average_triple_double * 5

            # 傷病因素調整分數
            team_total_games = 82
            games_played = pd.to_numeric(player_stats['G'], errors='coerce')
            injury_impact = games_played / team_total_games if team_total_games > 0 else 0
            adjusted_score = player_score * injury_impact
            no_injury_score = player_score

            # 存入快取
            fitness_cache[player_id] = (player_score, adjusted_score, injury_impact, no_injury_score, player_stats)

        # 檢查位置適合度
        if position not in player_stats['Pos'].split('-'):
            position_penalty += 5

        total_score += adjusted_score
        selected_players.append((player_stats['Player'], adjusted_score, player_stats['Pos'], injury_impact, no_injury_score))
        seen_players.add(player_id)

    return total_score - position_penalty, selected_players


# 初始化種群
def initialize_population():
    population = []
    for _ in range(600):  # 種群大小
        team = []
        selected_players = set()
        for position in positions_required:
            # 優先選擇得分能力排名前 50% 的球員
            candidates_pts = player_data[player_data['Pos'].str.contains(position, na=False)].sort_values('PTS', ascending=False)['Rk'].tolist()
            candidates_ast = player_data[player_data['Pos'].str.contains(position, na=False)].sort_values('AST', ascending=False)['Rk'].tolist()
            candidates_reb = player_data[player_data['Pos'].str.contains(position, na=False)].sort_values('ORB', ascending=False)['Rk'].tolist()
            # 合併多個特性候選人列表，避免重複
            candidates = list(set(candidates_pts[:30] + candidates_ast[:30] + candidates_reb[:30]))
            candidates = [player for player in candidates if player not in selected_players]
            if candidates:
                chosen_player = random.choice(candidates)
                team.append(chosen_player)
                selected_players.add(chosen_player)
        population.append(team)
    return population

# 傷病影響分析可視化
def visualize_injury_impact(selected_players):
    players = [p[0] for p in selected_players]
    games_played_ratios = [p[3] for p in selected_players]
    adjusted_scores = [p[1] for p in selected_players]
    no_injury_scores = [p[4] for p in selected_players]  # 取出 No Injury Score

    # 將 Games Played Ratio 調整到與 Adjusted Score 相同的數值範圍
    games_played_ratios_scaled = [x * 100 for x in games_played_ratios]

    # 設定柱狀圖的寬度
    bar_width = 0.25
    index = np.arange(len(players))  # 設定 x 軸位置

    # 繪製圖表
    plt.figure(figsize=(16, 7))

    plt.bar(index - bar_width, no_injury_scores, width=bar_width, label='No Injury Score', color='black')
    plt.bar(index, adjusted_scores, width=bar_width, label='Adjusted Score', color='purple')
    plt.bar(index + bar_width, games_played_ratios_scaled, width=bar_width, label='Games Played Ratio (Scaled)', color='orange')

    # 調整 x 軸標籤
    plt.xlabel('Player')
    plt.ylabel('Impact / Score')
    plt.title('Injury Impact on Player Scores (With No Injury Score)')
    plt.xticks(index, players, rotation=45, ha='right')

    plt.legend()
    plt.tight_layout()
    plt.show()


# 計算傷病影響
def calculate_injury_impact(player_stats, player_score):
    games_played = player_stats.get('G', 0)
    adjusted_score = player_score * (games_played / 82) if games_played > 0 else 0
    injury_impact = games_played / 82 if games_played > 0 else 0
    return adjusted_score, injury_impact

# 交叉操作
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)  # 避免在首尾進行交叉
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    # 移除子代中的重複球員
    child1 = repair_duplicates(child1)
    child2 = repair_duplicates(child2)

    return child1, child2

def repair_duplicates(team):
    seen = set()
    new_team = []
    for position, player_id in zip(positions_required, team):
        if player_id not in seen:
            new_team.append(player_id)
            seen.add(player_id)
        else:
            # 替換為適應度最高且符合多位置需求的候選球員
            candidates = player_data[player_data['Pos'].str.contains(position, na=False)]
            candidates = candidates[~candidates['Rk'].isin(seen)].sort_values('PTS', ascending=False)
            if not candidates.empty:
                replacement = candidates.iloc[0]  # 得分最高的候選人
                new_team.append(replacement['Rk'])
                seen.add(replacement['Rk'])
    return new_team

'''def print_player_scores():
    players_of_interest = ['Trae Young', 'Tyrese Haliburton']
    for player_name in players_of_interest:
        player_stats = player_data[player_data['Player'].str.contains(player_name, case=False)]
        if not player_stats.empty:
            player_id = player_stats.iloc[0]['Rk']
            score, _ = fitness_function([player_id])
            print(f"{player_name}: Fitness = {score:.2f}")
        else:
            print(f"{player_name}: Data not found")
'''
def log_missing_players(team):
    players_of_interest = ['Trae Young', 'Tyrese Haliburton', 'Anthony Davis']
    for player_name in players_of_interest:
        if player_name not in [player[0] for player in team]:
            print(f"Missing player: {player_name}")
def position_match(player_pos, target_pos):
    if not isinstance(player_pos, str):
        return False
    return target_pos in player_pos.split('-')

def local_search(team):
    # 找到球隊中fitness最低的球員
    lowest_score_index = -1
    lowest_score = float('inf')

    for i, player_id in enumerate(team):
        player_stats = player_data[player_data['Rk'] == player_id]
        if not player_stats.empty:
            player_stats = player_stats.iloc[0]
            # 計算Fitness分數
            player_score = (
                player_stats['FGA'] * scoring_system['FGA'] +
                player_stats['FG'] * scoring_system['FG'] +
                player_stats['FTA'] * scoring_system['FTA'] +
                player_stats['FT'] * scoring_system['FT'] +
                player_stats['3P'] * scoring_system['3P'] +
                player_stats['PTS'] * scoring_system['PTS'] +
                player_stats['ORB'] * scoring_system['OREB'] +
                player_stats['DRB'] * scoring_system['DRB'] +
                player_stats['AST'] * scoring_system['AST'] +
                player_stats['STL'] * scoring_system['STL'] +
                player_stats['BLK'] * scoring_system['BLK'] +
                player_stats['TOV'] * scoring_system['TOV'] +
                player_stats['PF'] * scoring_system['PF']
            )
            if player_score < lowest_score:
                lowest_score = player_score
                lowest_score_index = i

    # 替換fitness最低的球員
    if lowest_score_index != -1:
        position = positions_required[lowest_score_index]

        # 找該位置PTS前15名球員
        all_candidates = player_data.copy()
        all_candidates = all_candidates[all_candidates['Pos'].apply(lambda p: position_match(p, position))]
        all_candidates = all_candidates.sort_values('PTS', ascending=False).head(15)

        if not all_candidates.empty:
            # 在這15人中計算Fitness
            all_candidates['Fitness_Score'] = (
                all_candidates['FGA'] * scoring_system['FGA'] +
                all_candidates['FG'] * scoring_system['FG'] +
                all_candidates['FTA'] * scoring_system['FTA'] +
                all_candidates['FT'] * scoring_system['FT'] +
                all_candidates['3P'] * scoring_system['3P'] +
                all_candidates['PTS'] * scoring_system['PTS'] +
                all_candidates['ORB'] * scoring_system['OREB'] +
                all_candidates['DRB'] * scoring_system['DRB'] +
                all_candidates['AST'] * scoring_system['AST'] +
                all_candidates['STL'] * scoring_system['STL'] +
                all_candidates['BLK'] * scoring_system['BLK'] +
                all_candidates['TOV'] * scoring_system['TOV'] +
                all_candidates['PF'] * scoring_system['PF']
            )

            # 選Fitness最高的人替換
            best_candidate = all_candidates.sort_values('Fitness_Score', ascending=False).iloc[0]
            team[lowest_score_index] = best_candidate['Rk']

    return team


# 突變操作
def mutate(team, mutation_rate=0.2):
    for i in range(len(team)):
        if random.random() < mutation_rate:
            position = positions_required[i]
            current_player_id = team[i]

            # 只考慮符合當前位置需求的球員
            candidates = player_data[player_data['Pos'].str.contains(position, na=False)]
            candidates = candidates[~candidates['Rk'].isin(team)].sort_values('PTS', ascending=False)

            if not candidates.empty:
                # 隨機選擇候選球員，但依照分數排序
                new_player = candidates.sample(n=1)['Rk'].values[0]
                team[i] = new_player

    return team
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def analyze_results_ignore_order_with_plot(best_team, ideal_team):
    """
    分析算法結果與理想解的準確性，並繪製混淆矩陣
    """
    # 將 `best_team` 中的 `Rk` 轉換為球員名稱
    best_team_names = set(
        player_data[player_data['Rk'].isin(best_team)]['Player'].str.strip().str.lower()
    )

    # 將 `ideal_team` 轉換為小寫，確保名稱格式一致
    ideal_team_names = {name.strip().lower() for name in ideal_team}

    # 計算 TP, FP, FN
    tp = len(best_team_names & ideal_team_names)  # 正確選擇的球員
    fp = len(best_team_names - ideal_team_names)  # 錯誤選擇的球員
    fn = len(ideal_team_names - best_team_names)  # 未選到的正確球員

    # 打印分析結果
    print("=== 分析結果 ===")
    print(f"True Positives (TP): {tp} - 正確選擇的球員數量")
    print(f"False Positives (FP): {fp} - 錯誤選擇的球員數量")
    print(f"False Negatives (FN): {fn} - 未選到的正確球員數量")

    # 顯示詳細信息
    print("\n=== 錯誤選擇的球員 ===")
    print(best_team_names - ideal_team_names)

    print("\n=== 未選到的正確球員 ===")
    print(ideal_team_names - best_team_names)

    # 計算準確率和召回率
    accuracy = tp / len(ideal_team_names) if len(ideal_team_names) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n=== 評估指標 ===")
    print(f"Accuracy (準確率): {accuracy:.2f}")
    print(f"Precision (精確率): {precision:.2f}")
    print(f"Recall (召回率): {recall:.2f}")

    # 混淆矩陣的構建
    y_true = [1 if name in ideal_team_names else 0 for name in player_data['Player'].str.strip().str.lower()]
    y_pred = [1 if name in best_team_names else 0 for name in player_data['Player'].str.strip().str.lower()]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    # 繪製混淆矩陣
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Selected', 'Not Selected'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

def mutate_elite(elite, mutation_rate=0.1):
    """
    對精英個體進行小幅變異
    """
    new_elite = []
    for team in elite:
        if random.random() < mutation_rate:
            new_elite.append(mutate(team, mutation_rate=0.1))  # 對精英個體進行突變
        else:
            new_elite.append(team)
    return new_elite

def analyze_injury_impact(selected_players):
    print("\n=== 傷病影響分析 ===")
    for player, adjusted_score, position, injury_impact, no_injury_score in selected_players:
        print(f"{player} ({position}): Adjusted Score = {adjusted_score:.2f}, No Injury Score = {no_injury_score:.2f}, Games Played Ratio = {injury_impact:.2%}")
    visualize_injury_impact(selected_players)



# 遺傳算法主流程
def genetic_algorithm():
    start_time = time.time()  # ⏱️ 開始計時
    population = initialize_population()
    best_fitness_history = []

    for generation in range(150):  # 增加世代數
        fitness_scores = []
        team_selections = []

        for team in population:
            score, selected_players = fitness_function(team)
            fitness_scores.append(score)
            team_selections.append(selected_players)

        # 保留精英
        elite_count = max(2, int(len(population) * 0.1))  # 保留 10% 精英
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
        elites = [population[i] for i in elite_indices]  # 保留精英個體
        elites = mutate_elite(elites)  # 對精英進行小幅變異

        # 計算多樣性
        diversity = np.std(fitness_scores)
        print(f"Generation {generation + 1}: Population Diversity = {diversity}")

        # 重新初始化部分種群（避免早熟收斂）
        if diversity < 1e-3:  # 如果多樣性過低
            for i in range(len(population) // 10):  # 替換 10% 種群
                population[i] = initialize_population()[0]

        # 當前世代最佳解
        best_index = fitness_scores.index(max(fitness_scores))
        best_team = population[best_index]
        best_fitness = fitness_scores[best_index]
        best_fitness_history.append(best_fitness)

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
        for player, score, position, injury_impact, no_injury_score in team_selections[best_index]:
            print(f"  {player} ({position}): Adjusted Score = {score:.2f}, No Injury Score = {no_injury_score:.2f}, Games Played Ratio = {injury_impact:.2%}")

        # 提前停止條件：檢查是否滿足目標球員組合

        # 提前停止條件：檢查是否滿足目標球員組合
        best_team_names = set(
            player_data[player_data['Rk'].isin(best_team)]['Player'].str.strip().str.lower()
        )
        if target_players.issubset(best_team_names):
            print(f"Target team found at generation {generation + 1} with fitness {best_fitness}. Stopping early.")
            break  # 提前結束循環

        # 選擇下一代
        next_generation = elites.copy()
        for _ in range((len(population) - elite_count) // 2):
            parent1 = random.choices(population, weights=fitness_scores, k=1)[0]
            parent2 = random.choices(population, weights=fitness_scores, k=1)[0]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            child1 = local_search(child1)
            child2 = local_search(child2)
            # 檢查子代是否優於父代
            if fitness_function(child1)[0] < fitness_function(parent1)[0]:
                child1 = parent1
            if fitness_function(child2)[0] < fitness_function(parent2)[0]:
                child2 = parent2

            # 添加到下一代
            next_generation.extend([child1, child2])

        population = next_generation
    end_time = time.time()  # ⏱️ 結束計時
    total_time = end_time - start_time
    print(f"⏱️ Total execution time: {total_time:.2f} seconds")

    

    return best_team, best_fitness, total_time, generation + 1,best_fitness_history
# 多輪實驗放在外面！
num_runs = 5
total_time = 0
total_generations = 0
all_histories = []
for run in range(num_runs):
    print(f"\n🔁 Run {run + 1}")
    fitness_cache.clear()
    best_team, best_fitness, exec_time, generations,best_fitness_history = genetic_algorithm()
    total_time += exec_time
    total_generations += generations
    all_histories.append(best_fitness_history) 
avg_time = total_time / num_runs
avg_generations = total_generations / num_runs

print(f"\n✅ Average over {num_runs} runs:")
print(f"⏱️ Average Time: {avg_time:.2f} seconds")
print(f"📈 Average Generations: {avg_generations:.2f}")
# 理想解
ideal_team = [
    "james harden", "shai gilgeous-alexander", "anthony edwards", "trae young",
    "cade cunningham","jayson tatum", "giannis antetokounmpo", "karl-anthony towns",
    "lebron james","nikola jokic" 
]

# 執行遺傳算法前清空快取
fitness_cache.clear()  # 確保每次實驗開始時沒有舊的資料干擾
np.save('normal_this_end.npy', np.array(best_fitness_history))
'''# 執行遺傳算法
best_team, best_fitness, exec_time, generations = genetic_algorithm()
# 對最佳球隊重新計算適應度以獲取完整數據'''
_, selected_players = fitness_function(best_team)
# 繪製收斂曲線
plt.plot(range(len(best_fitness_history)), best_fitness_history, marker='o')
plt.title("Convergence Curve")
plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()
# 對結果進行分析並繪製混淆矩陣
analyze_results_ignore_order_with_plot(best_team, ideal_team)
analyze_injury_impact(selected_players)

