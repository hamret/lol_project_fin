# feature_builder_challenger_full.py
import numpy as np

LANES = ["TOP", "JUNGLE", "MID", "BOTTOM"]


class FeatureBuilderChallengerFull:

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # 1) timeline 전체 이벤트 파싱 (오브젝트 / 타워 / 킬 / 전령 등)
    # ---------------------------------------------------------
    def parse_events(self, timeline):
        events = []

        for frame in timeline["info"]["frames"]:
            if "events" not in frame:
                continue
            for ev in frame["events"]:
                events.append(ev)

        return events

    # ---------------------------------------------------------
    # 2) 오브젝트 정보 추출
    # ---------------------------------------------------------
    def extract_objective_features(self, events):
        drake_counts = {"infernal": 0, "ocean": 0, "mountain": 0, "cloud": 0}
        herald_count = 0
        herald_summon = 0
        tower_kills = 0

        for ev in events:

            # 용 처치
            if ev["type"] == "ELITE_MONSTER_KILL" and ev["monsterType"] == "DRAGON":
                d = ev["monsterSubType"].lower()
                if d in drake_counts:
                    drake_counts[d] += 1

            # 전령 처치
            if ev["type"] == "ELITE_MONSTER_KILL" and ev["monsterType"] == "RIFTHERALD":
                herald_count += 1

            # 전령 소환
            if ev["type"] == "HERALD_SUMMONED":
                herald_summon += 1

            # 타워 파괴
            if ev["type"] == "BUILDING_KILL" and ev["buildingType"] == "TOWER_BUILDING":
                tower_kills += 1

        # 오브젝트 점수를 하나로 합성
        obj_score = (
            drake_counts["infernal"] * 3 +
            drake_counts["mountain"] * 2.5 +
            drake_counts["ocean"] * 2 +
            drake_counts["cloud"] * 1.5 +
            herald_count * 3 +
            herald_summon * 2 +
            tower_kills * 1.5
        )

        return {
            "drake": drake_counts,
            "herald": herald_count,
            "heraldSummon": herald_summon,
            "towerKills": tower_kills,
            "objectiveScore": obj_score
        }

    # ---------------------------------------------------------
    # 3) 정글 개입 측정 (lane proximity score)
    # ---------------------------------------------------------
    def extract_jungle_pressure(self, match, timeline):
        frames = timeline["info"]["frames"]

        # 정글러 participantId
        jungle_blue = None
        jungle_red = None
        for p in match["info"]["participants"]:
            if p["teamPosition"] == "JUNGLE":
                if p["teamId"] == 100:
                    jungle_blue = str(p["participantId"])
                else:
                    jungle_red = str(p["participantId"])

        pressure = {l: 0 for l in LANES}

        for minute in range(1, min(16, len(frames))):
            frame = frames[minute]
            pf = frame["participantFrames"]

            # 정글러 좌표
            jx_b = pf[jungle_blue]["position"]["x"] if jungle_blue else None
            jy_b = pf[jungle_blue]["position"]["y"] if jungle_blue else None
            jx_r = pf[jungle_red]["position"]["x"] if jungle_red else None
            jy_r = pf[jungle_red]["position"]["y"] if jungle_red else None

            for p in match["info"]["participants"]:
                pos = p["teamPosition"]
                if pos == "MIDDLE": pos = "MID"
                if pos == "UTILITY": pos = "BOTTOM"
                if pos not in LANES:
                    continue

                pid = str(p["participantId"])
                laner = pf[pid]["position"]
                lx, ly = laner["x"], laner["y"]

                # 거리 계산
                if jx_b is not None:
                    dist_b = ((lx - jx_b)**2 + (ly - jy_b)**2)**0.5
                    pressure[pos] += max(0, (8000 - dist_b) / 8000)

                if jx_r is not None:
                    dist_r = ((lx - jx_r)**2 + (ly - jy_r)**2)**0.5
                    pressure[pos] += max(0, (8000 - dist_r) / 8000)

        return pressure

    # ---------------------------------------------------------
    # 4) 라인전 피처 + 정글 개입 + 오브젝트 + 팀 템포
    # ---------------------------------------------------------
    def extract_timeseries(self, match, timeline):

        frames = timeline["info"]["frames"]
        events = self.parse_events(timeline)
        obj = self.extract_objective_features(events)
        pressure = self.extract_jungle_pressure(match, timeline)

        # lane pid 매핑
        lane_pid = {}
        for p in match["info"]["participants"]:
            pos = p["teamPosition"]
            if pos == "MIDDLE": pos = "MID"
            if pos == "UTILITY": pos = "BOTTOM"
            if pos not in LANES:
                continue
            lane_pid[pos] = str(p["participantId"])

        ts = {l: [] for l in LANES}

        # 0~15분 프레임 기반 feature 생성
        for minute in range(0, 16):
            frame = frames[minute]["participantFrames"]

            # 각 lane 피처 계산
            for lane in LANES:
                pid = lane_pid[lane]
                pf = frame[pid]

                gold = pf["totalGold"]
                xp = pf["xp"]
                lvl = pf["level"]
                cs = pf["minionsKilled"]
                jg = pf["jungleMinionsKilled"]
                dmg = pf.get("damageStats", {}).get("totalDamageDealtToChampions", 0)

                # 라인 격차
                gold_diff, xp_diff = self._lane_diffs(match, frame, lane)

                # 성장률
                if minute > 0:
                    prev_pf = frames[minute - 1]["participantFrames"][pid]
                    gold_rate = gold - prev_pf["totalGold"]
                    xp_rate = xp - prev_pf["xp"]
                else:
                    gold_rate = 0
                    xp_rate = 0

                # power score
                power = gold + xp * 0.7

                # 팀 템포
                tempo = (obj["objectiveScore"] + pressure[lane]) / 10

                # 개별 feature 묶기
                ts[lane].append([
                    gold, xp, lvl, cs, jg, dmg,
                    gold_diff, xp_diff,
                    gold_rate, xp_rate,
                    power, tempo,
                    pressure[lane],
                    obj["objectiveScore"],
                    obj["drake"]["infernal"],
                    obj["drake"]["ocean"],
                    obj["drake"]["mountain"],
                    obj["drake"]["cloud"],
                    obj["herald"],
                    obj["towerKills"]
                ])

        for lane in LANES:
            ts[lane] = np.array(ts[lane])

        return ts

    # ---------------------------------------------------------
    # 라인별 골드/XP 격차 계산
    # ---------------------------------------------------------
    def _lane_diffs(self, match, pf, target_lane):
        ally = enemy = None

        for p in match["info"]["participants"]:
            pos = p["teamPosition"]
            if pos == "MIDDLE": pos = "MID"
            if pos == "UTILITY": pos = "BOTTOM"
            if pos != target_lane:
                continue

            pid = str(p["participantId"])
            gold = pf[pid]["totalGold"]
            xp = pf[pid]["xp"]

            if p["teamId"] == 100:
                ally = (gold, xp)
            else:
                enemy = (gold, xp)

        if ally is None or enemy is None:
            return 0, 0

        return ally[0] - enemy[0], ally[1] - enemy[1]
