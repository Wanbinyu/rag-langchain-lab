from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SOURCE_ROOT = "graham-essays"


def source(name: str) -> str:
    return f"{SOURCE_ROOT}/{name}.md"


# These are manually curated gold-source labels.  Several questions deliberately
# have more than one source because the reference answer combines principles
# from multiple essays.
SOURCES: dict[int, list[str]] = {
    1: [source("151_how-to-get-startup-ideas")],
    2: [source("153_do-things-that-don-t-scale")],
    3: [source("069_the-18-mistakes-that-kill-startups"), source("105_startups-in-13-sentences")],
    4: [source("115_maker-s-schedule-manager-s-schedule")],
    5: [source("028_how-to-make-wealth")],
    6: [source("105_startups-in-13-sentences"), source("124_apple-s-mistake")],
    7: [source("169_default-alive-or-default-dead")],
    8: [source("105_startups-in-13-sentences"), source("069_the-18-mistakes-that-kill-startups")],
    9: [source("105_startups-in-13-sentences"), source("151_how-to-get-startup-ideas"), source("153_do-things-that-don-t-scale")],
    10: [source("056_how-to-do-what-you-love"), source("218_how-to-do-great-work")],
    11: [source("116_ramen-profitable"), source("105_startups-in-13-sentences")],
    12: [source("161_how-to-be-an-expert-in-a-changing-world"), source("105_startups-in-13-sentences")],
    13: [source("105_startups-in-13-sentences"), source("151_how-to-get-startup-ideas")],
    14: [source("170_write-like-you-talk")],
    15: [source("143_frighteningly-ambitious-startup-ideas")],
    16: [source("153_do-things-that-don-t-scale"), source("105_startups-in-13-sentences")],
    17: [source("105_startups-in-13-sentences"), source("153_do-things-that-don-t-scale")],
    18: [source("149_startup-growth"), source("153_do-things-that-don-t-scale")],
    19: [source("105_startups-in-13-sentences")],
    20: [source("105_startups-in-13-sentences"), source("116_ramen-profitable")],
    21: [source("069_the-18-mistakes-that-kill-startups")],
    22: [source("105_startups-in-13-sentences"), source("149_startup-growth")],
    23: [source("105_startups-in-13-sentences"), source("218_how-to-do-great-work")],
    24: [source("125_organic-startup-ideas"), source("151_how-to-get-startup-ideas")],
    25: [source("105_startups-in-13-sentences"), source("151_how-to-get-startup-ideas")],
    26: [source("028_how-to-make-wealth"), source("105_startups-in-13-sentences")],
    27: [source("105_startups-in-13-sentences"), source("153_do-things-that-don-t-scale")],
    28: [source("169_default-alive-or-default-dead"), source("105_startups-in-13-sentences")],
    29: [source("115_maker-s-schedule-manager-s-schedule")],
    30: [source("115_maker-s-schedule-manager-s-schedule")],
    31: [source("218_how-to-do-great-work"), source("056_how-to-do-what-you-love")],
    32: [source("218_how-to-do-great-work"), source("056_how-to-do-what-you-love")],
    33: [source("220_the-best-essay"), source("218_how-to-do-great-work")],
    34: [source("220_the-best-essay")],
    35: [source("218_how-to-do-great-work")],
    36: [source("218_how-to-do-great-work")],
    37: [source("218_how-to-do-great-work"), source("180_the-bus-ticket-theory-of-genius")],
    38: [source("180_the-bus-ticket-theory-of-genius")],
    39: [source("170_write-like-you-talk"), source("200_write-simply")],
    40: [source("170_write-like-you-talk")],
    41: [source("105_startups-in-13-sentences"), source("039_how-to-start-a-startup")],
    42: [source("105_startups-in-13-sentences"), source("153_do-things-that-don-t-scale")],
    43: [source("105_startups-in-13-sentences"), source("039_how-to-start-a-startup")],
    44: [source("105_startups-in-13-sentences")],
    45: [source("105_startups-in-13-sentences"), source("069_the-18-mistakes-that-kill-startups")],
    46: [source("153_do-things-that-don-t-scale"), source("105_startups-in-13-sentences")],
    47: [source("153_do-things-that-don-t-scale")],
    48: [source("151_how-to-get-startup-ideas"), source("125_organic-startup-ideas")],
    49: [source("105_startups-in-13-sentences"), source("218_how-to-do-great-work")],
    50: [source("105_startups-in-13-sentences"), source("218_how-to-do-great-work")],
    51: [source("115_maker-s-schedule-manager-s-schedule")],
    52: [source("115_maker-s-schedule-manager-s-schedule")],
    53: [source("218_how-to-do-great-work"), source("104_keep-your-identity-small")],
    54: [source("218_how-to-do-great-work"), source("056_how-to-do-what-you-love")],
    55: [source("218_how-to-do-great-work"), source("056_how-to-do-what-you-love")],
    56: [source("218_how-to-do-great-work")],
    57: [source("218_how-to-do-great-work"), source("180_the-bus-ticket-theory-of-genius")],
    58: [source("194_how-to-think-for-yourself"), source("187_being-a-noob")],
    59: [source("169_default-alive-or-default-dead")],
    60: [source("105_startups-in-13-sentences"), source("218_how-to-do-great-work")],
    61: [source("105_startups-in-13-sentences"), source("153_do-things-that-don-t-scale")],
    62: [source("218_how-to-do-great-work")],
    63: [source("218_how-to-do-great-work"), source("051_ideas-for-startups")],
    64: [source("105_startups-in-13-sentences"), source("039_how-to-start-a-startup"), source("019_design-and-research")],
    65: [source("151_how-to-get-startup-ideas"), source("143_frighteningly-ambitious-startup-ideas"), source("218_how-to-do-great-work")],
}


KEYWORDS: dict[int, list[str]] = {
    1: ["发现问题", "自己想要", "自己能做出来"],
    2: ["无法规模化", "手动", "招募用户"],
    3: ["用户真正想要", "users want", "错误"],
    4: ["Maker", "Manager", "连续时间"],
    5: ["人们想要", "让世界变得更好", "财富"],
    6: ["尽快上线", "Launch", "真实反馈"],
    7: ["Default Alive", "Default Dead", "盈亏平衡"],
    8: ["联合创始人", "cofounders", "重要"],
    9: ["少量", "热爱", "口碑"],
    10: ["工作和生活", "融合", "兴趣"],
    11: ["Ramen", "基本生活", "拉面"],
    12: ["用户", "问题领域", "专家"],
    13: ["想法", "迭代", "执行"],
    14: ["Write Like You Talk", "简单", "清晰"],
    15: ["害怕", "野心", "风险"],
    16: ["早期用户", "亲自", "反馈"],
    17: ["热爱", "少量", "强烈"],
    18: ["增长", "项目", "核心"],
    19: ["衡量", "指标", "方向"],
    20: ["少花钱", "Ramen", "低成本"],
    21: ["用户真正想要", "错误", "根本"],
    22: ["持续使用", "增长", "口碑"],
    23: ["用户反馈", "放弃", "方向"],
    24: ["有机", "自然", "问题"],
    25: ["用户", "变化", "迭代"],
    26: ["创造财富", "人们真正想要", "世界"],
    27: ["满意度", "少量", "扩展"],
    28: ["Default Dead", "融资", "削减开支"],
    29: ["会议", "打断", "创造者"],
    30: ["连续", "打断", "效率"],
    31: ["兴趣", "声望", "伟大工作"],
    32: ["兴趣", "坚持", "成果"],
    33: ["问题", "发现", "回答"],
    34: ["问题", "未知", "发现"],
    35: ["前沿", "未解决", "领域"],
    36: ["兴趣", "天赋", "贡献"],
    37: ["天赋", "努力", "好奇心"],
    38: ["Bus Ticket", "专注", "兴趣"],
    39: ["简单", "说话", "写作"],
    40: ["像说话一样", "自然", "理解"],
    41: ["理解用户", "想要", "早期"],
    42: ["迭代", "用户反馈", "想法"],
    43: ["理解用户", "财富", "改善"],
    44: ["联合创始人", "上线", "Ramen"],
    45: ["联合创始人", "信任", "合作"],
    46: ["早期客户", "反馈", "快速"],
    47: ["规模化", "用户", "早期"],
    48: ["自己想要", "自己能做", "价值"],
    49: ["失败", "反馈", "放弃"],
    50: ["用户数据", "反馈", "情绪"],
    51: ["大块", "会议", "创造者"],
    52: ["连续", "深度", "打断"],
    53: ["社会压力", "声望", "兴趣"],
    54: ["自己想做", "期望", "兴趣"],
    55: ["兴奋", "贡献", "价值"],
    56: ["兴趣", "前沿", "持续"],
    57: ["好奇心", "伟大工作", "驱动"],
    58: ["天真", "框架", "可能性"],
    59: ["Default Alive", "烧钱", "盈亏平衡"],
    60: ["专注", "用户", "不确定性"],
    61: ["少数用户", "满意", "口碑"],
    62: ["竞争", "用户", "产品"],
    63: ["竞争", "用户", "产品"],
    64: ["产品", "用户", "技术"],
    65: ["兴趣", "用户", "成长"],
}


def parse_attachment(path: Path) -> list[tuple[int, str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        match = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", line)
        if not match:
            continue
        if index + 1 >= len(lines):
            raise ValueError(f"第 {match.group(1)} 题缺少答案")
        answer_line = lines[index + 1].strip()
        if not answer_line.startswith("答案"):
            raise ValueError(f"第 {match.group(1)} 题的下一行不是答案")
        parsed.append((int(match.group(1)), match.group(2), answer_line[3:].strip()))
    if len(parsed) != 85 or [item[0] for item in parsed] != list(range(1, 86)):
        raise ValueError(f"期望连续的 1-85 题，实际解析到 {len(parsed)} 题")
    return parsed


def category(number: int) -> str:
    if number >= 66:
        return "out_of_scope"
    if number in {4, 29, 30, 51, 52}:
        return "maker_schedule"
    if number in {14, 39, 40}:
        return "writing"
    if number in {31, 32, 33, 34, 35, 36, 37, 38, 53, 54, 55, 56, 57, 58}:
        return "great_work"
    if number in {5, 7, 11, 18, 19, 20, 26, 28, 59}:
        return "startup_finance"
    return "startup_execution"


def build_dataset(input_path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for number, question, answer in parse_attachment(input_path):
        positive = number <= 65
        cases.append(
            {
                "id": f"graham-{number:03d}",
                "category": category(number),
                "question": question,
                "expected_sources": SOURCES.get(number, []),
                "expected_keywords": KEYWORDS.get(number, []) if positive else [],
                "expect_abstention": not positive,
                "reference_answer": answer,
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="从问答附件生成 Graham essays 评测集")
    parser.add_argument("input", type=Path, help="用户提供的 pasted-text.txt")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("eval/questions_graham_85.jsonl"),
    )
    args = parser.parse_args()
    cases = build_dataset(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "cases": len(cases), "positive": 65, "negative": 20}, ensure_ascii=False))


if __name__ == "__main__":
    main()
