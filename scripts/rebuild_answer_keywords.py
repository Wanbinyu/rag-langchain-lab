"""Build the auditable keyword rubric for the 150-question Graham set.

The rubric is deliberately kept outside ``data/`` and the vector store.  A
required group represents one concept that a generated answer should cover;
terms inside a group are interchangeable Chinese/English expressions.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "eval" / "questions_graham_150.jsonl"
OUTPUT_PATH = ROOT / "eval" / "answer_keywords_graham_150.json"


# Format: question id, then required groups separated by ``||``.  Alternatives
# in a group are separated by ``|``.  Keeping the rubric in a compact table
# makes review and later edits easier than hiding it in scoring code.
RUBRIC = r"""
001|发现身边的真实问题|发现真实问题|发现问题||创始人自己想要|自己想要||自己能做出来|自己能够做出来|能做出来
002|无法规模化|不规模化|unscalable||亲自服务用户|手动完成|招募用户||少数用户真正喜欢|少量用户喜欢||逐步规模化|找到可规模化
003|用户真正想要|人们真正想要|users want||没有做出用户想要|not making something users want
004|Maker's Schedule|Maker’s Schedule|Maker||大块连续时间|连续时间|深度工作||Manager's Schedule|Manager’s Schedule|Manager||会议切割成小块|时间被会议切割|小块时间
005|财富是人们想要的东西|人们想要的东西|wealth is what people want||让世界变得更好|改善世界||创造财富|create wealth||不是转移财富|财富转移
006|尽快上线|快速上线|Launch Fast|launch fast||早期想法会变|想法几乎都会变||真实用户反馈|真实反馈|user feedback||快速迭代|iterate quickly
007|Default Alive|资金能支撑到盈亏平衡|现有资金达到盈亏平衡||Default Dead|资金无法支撑到盈亏平衡||立刻行动|削减开支|融资
008|联合创始人|cofounder|cofounders||非常重要|关键因素||关系导致失败|创业失败
009|少数热爱产品的用户|少量热爱用户|少而精||清晰反馈|强烈正反馈||口碑传播|口碑||大量无所谓用户|模糊信号
010|工作和生活融合|工作生活融合||不严格分开|不是严格分开||热爱工作|真正感兴趣
011|Ramen Profitable|Ramen||覆盖创始人基本生活开销|支付创始人生活费用||不依赖投资人的钱|不再烧投资人的钱
012|用户和问题领域的专家|用户领域专家||创业方法论专家|创业专家|方法论
013|想法最终会改变|想法大变||发现问题|问题洞察||持续迭代|持续调整||执行和洞察|执行能力
014|Write Like You Talk|像说话一样写作|口语化写作||简单清晰直接|简单语言|清晰表达
015|有野心的想法|野心想法||感到害怕|有点害怕||改变自我认知|身份认知||看起来困难|过于困难
016|亲自服务早期用户|亲自服务用户||理解用户需求和痛点|用户需求|用户痛点||真实反馈|用户反馈||快速改进产品|改进产品
017|增长是核心特征|增长|growth||没有明显增长|普通项目
018|你做你衡量的东西|you are what you measure||优化正在衡量的指标|优化指标||指标选错导致方向错误|错误方向|错误指标
019|少花钱|低成本|节省开支||Ramen Profitable|Ramen||尽快达到|快速达到
020|最危险的错误|创业致命错误||没有做出用户真正想要|用户想要|users want
021|用户真正喜欢|用户喜爱||持续使用|留存|持续使用||自然增长|口碑增长
022|不要轻易放弃|坚持||根据真实用户反馈判断|用户反馈||避免盲目坚持|不要盲目
023|有机创业想法|自然生长的想法|organic ideas||创始人自己遇到的问题|自己遇到的问题
024|真正接触用户|接触用户||最初假设被证明错误|假设错误||根据反馈演进|反馈调整|演进
025|创造财富|create wealth||人们真正想要的东西|用户想要
026|用户满意度优先|用户满意||少数人非常满意|少数用户||再扩展|扩大规模
027|Default Dead||尽快融资|融资
028|会议打断创造者|会议打断|meetings interrupt||深度工作|连续时间||减少会议|少开会议
029|大块连续时间|连续时间||进入心流|深度工作|flow||频繁打断降低效率|打断效率
030|真正感兴趣的事情|兴趣优先||声望随后而来|不为声望|声望
031|长期坚持|持续努力||有价值的成果|重要成果
032|好的问题接近发现|问题接近发现||回答过程中不断生长|问题生长|不断深入
033|提出正确的问题|正确问题||指向重要未知领域|未知领域
034|选择领域|选定领域||学习到接近前沿|领域前沿|学到前沿||探索未解决问题|未解决的问题
035|兴趣|真正感兴趣||天赋|才能||贡献空间|做出重要贡献
036|天赋重要|天赋||持续努力|长期努力||好奇心|保持好奇
037|Bus Ticket Theory of Genius|Bus Ticket Theory||极度专注的兴趣|深度兴趣|专注||伟大工作的基础|伟大工作
038|简单直接的语言|简单语言||避免复杂和装腔作势|避免复杂|装腔作势
039|像说话一样写作|口语化||更自然更容易理解|自然|容易理解||接近真实思考|真实思考
040|理解用户|用户理解||做出用户真正想要|用户想要
041|真实用户反馈|用户反馈||快速迭代|快速改进||最初想法会改变|想法改变
042|改善用户生活|改善生活||理解用户需求|理解需求||创造财富|财富
043|选好联合创始人|Pick good cofounders|good cofounders||快速上线|Launch Fast||让想法演进|idea evolves||理解用户|用户理解||少数用户爱上产品|用户爱上
044|信任|相互信任||长期合作|长期合作关系||关系质量影响成败|合作成败
045|认真倾听客户反馈|倾听反馈||快速响应|快速回应||产品方向指引|产品方向
046|不规模的方式|Do Things That Don’t Scale|Do Things That Don't Scale||深入理解用户|理解用户||找到模式后规模化|再规模化
047|创始人自己想要|自己想要||自己能做|能做出来||别人未充分意识到价值|未意识到价值
048|失败是常态|失败常态||从中学习|学习||根据反馈调整方向|反馈调整
049|真实用户数据|用户数据||真实反馈|用户反馈||数据驱动决策|决策
050|保护大块连续时间|大块连续时间||避免会议和琐事切割|会议切割|琐事切割
051|进入深度工作状态|深度工作||频繁切换降低质量|切换降低效率|降低质量
052|社会期望和声望|社会压力|声望||忽略真正兴趣|忽略兴趣||阻碍伟大工作|伟大工作受阻
053|做自己真正想做的事情|自己想做||不迎合外部期望|迎合期望
054|让自己兴奋|兴趣||做出重要贡献|重要贡献||有价值的工作|有价值
055|真正感兴趣的领域|感兴趣领域||持续深入|深入||接近或拓展前沿|拓展前沿
056|好奇心|curiosity||伟大工作的驱动力|伟大工作
057|保持天真|天真||不受既有框架束缚|不被框架束缚||提出新问题|新问题
058|当前烧钱速度|burn rate|烧钱速度||现有资金支撑到盈亏平衡|资金支撑盈亏平衡||Default Alive
059|专注用户和产品|专注用户|专注产品||减少分心|少分心||接受不确定性|不确定性
060|少数用户|少量用户||清晰反馈|强烈反馈||形成口碑|口碑||大量无所谓用户|模糊信号
061|不应过度关注竞争对手|不过度关注竞争||专注自己的用户和产品|专注用户产品
062|不应过度关注竞争对手|不过度关注竞争|竞争对手
063|产品解决用户问题|解决用户问题||技术只是手段|技术手段
064|让创始人兴奋|创始人兴趣||改善用户生活|改善生活||大成长空间|成长空间
065|启动快|快速启动|Launch Fast||更快获得反馈|快速反馈||避免活在想象中|脱离想象
066|错误的指标|选错指标||错误方向优化|错误方向|优化错误
067|少数用户|少量用户||强烈正面情感|热爱产品|爱上产品||传播意愿|口碑传播
068|好的拖延和坏的拖延|好拖延|坏拖延||把时间花在更重要的事情上|重要事情
069|把想法写成文字|写下来||严格检验想法|检验想法||发现不完整之处|不完整
070|写作过程中|写下来||发现或完善想法|完善想法|形成想法
071|超预期客户服务|客户服务||客户忠诚|忠诚||口碑|口碑传播
072|专注核心|核心||减少不必要的活动|减少干扰|不必要活动
073|合作或交易最终落空|交易落空|交易黄||创业不确定性|不确定||做好心理准备|心理准备
074|不要轻易气馁|不气馁||关注小进展|小进展||用户反馈|反馈
075|根据反馈调整|反馈调整||优化最初产品或方向|优化产品|方向调整||让想法演进|想法演进
076|花费少|低成本||延长生存时间|延长跑道||降低融资依赖|减少融资依赖||保持灵活性|灵活
077|改善用户生活|用户生活改善||创造财富|财富||产品价值|价值
078|不为了声望|不为声望||兴趣优先|真正兴趣
079|选择领域|选择领域||学到前沿|领域前沿||发现缺口|发现缺口||探索缺口|探索问题
080|长期持续努力|持续努力||短期爆发|短期冲刺||积累成果|长期积累
081|接受不确定性是常态|不确定性常态||专注可控的用户和产品|可控用户产品
082|看起来困难|过于困难||改变自我认知|身份认知||有野心的想法|野心
083|早期用户|早期客户||直接体验产品|直接体验||真实改进方向|真实反馈|改进方向
084|新号|新账号||不要短时间大量上架|不要大量上架||保持自然节奏|自然节奏
085|口语化表达|像说话一样||简单直接|简单语言||容易理解|易理解
106|互补|互补能力||信任|相互信任||长期合作|长期合作关系||关系质量至关重要|关系质量
107|启动后想法演进|想法演进||现实反馈揭示不足|现实反馈|反馈||必须调整|调整方向
108|超预期客户服务|客户服务||用户忠诚|忠诚||口碑传播|口碑
109|关注小进展|小进展||用户反馈|反馈||不被短期挫折打倒|不要气馁
110|用户参与度|参与度||留存|用户留存||真实使用|真实使用||虚荣指标|虚荣指标
111|少花钱|低成本||延长跑道|生存时间||减少融资压力|融资压力||保持决策灵活性|灵活
112|Ramen Profitable|Ramen||开始自我造血|自我造血||降低生存风险|生存风险
113|指标会引导行为|指标引导行为||选错指标优化错误方向|错误指标|错误方向
114|过多会议|会议||无关项目|不必要项目||过度关注竞争对手|关注竞争
115|交易落空|交易黄||创业环境不确定|不确定性||承诺无法兑现|无法兑现
116|大块连续时间|连续时间||不被打断|不打断|深度工作
117|真正感兴趣|兴趣||更容易坚持并深入|坚持|深入
118|社会期望|外部期望||安全或有声望的路径|安全路径|声望路径||真正热情的方向|热爱方向
119|学到领域前沿|领域前沿||未被充分解决的问题|未解决问题|缺口
120|长期积累|持续积累||不是短期高强度|短期爆发
121|天真|保持天真||经验者忽略的可能性|被经验者忽略||新可能性|创新可能
122|用户热爱|用户喜欢||自然增长|自然增长
123|不规模的事情|Do Things That Don’t Scale|不规模||深入理解用户|理解用户||可规模化路径|规模化路径
124|真正使用|用户使用||喜欢|用户喜爱||愿意推荐|推荐
125|快速反馈|反馈循环||加速学习|快速学习||快速迭代|迭代
126|把模糊想法具体化|具体化想法||暴露逻辑漏洞|逻辑漏洞
127|减少会议|少开会议||批量处理事务|批量事务||设置不被打断的时段|不被打断|连续时间
128|令人不安|不安||挑战认知或能力边界|挑战认知||有野心的想法|野心
129|不同技能和视角|互补技能||覆盖更多问题|覆盖问题||减少盲点|盲点
130|失败产品版本|失败||学习机会|学习||根据反馈快速改进|反馈改进
131|创造新增价值|新增价值|创造财富||转移只是重新分配|财富转移|重新分配
132|用户需求是根本|用户需求||创业理论只是辅助|理论辅助
133|更早验证假设|快速验证|验证假设||避免错误方向投入|错误方向|减少投入
134|口碑传播|口碑||低成本获取新用户|低成本获客||验证产品吸引力|产品吸引力
135|兴趣|真正感兴趣||持续投入|长期投入||贡献空间|做出贡献
136|Maker|Maker's Schedule|Maker’s Schedule||打断心流|打断深度工作|心流打断||半天或一天效率丧失|效率丧失
137|写作过程|写下来||新的连接和洞察|新连接|洞察
138|可控进展|进展||用户信号|用户反馈||外部噪音|外部干扰
139|低成本运营|低成本||更高生存概率|生存概率||更强谈判地位|谈判地位
140|反思指标|检查指标||用户价值|用户价值||公司健康|公司健康
141|长期专注|长期深入||看到别人看不到的细节|独特细节||发现机会|机会
142|满意用户|用户满意||主动推荐|用户推荐||口碑飞轮|飞轮|增长
143|亲自做很多事|亲自做||直接接触现实|接触现实||加速学习和决策|学习决策
144|简单语言|简单表达||读完并理解|容易理解|读者理解||传播|传播效果
145|反馈指导探索|反馈||探索方向|探索||避免无目的发散|避免发散
"""


def parse_rubric() -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = {}
    for raw_line in RUBRIC.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|", 1)
        if len(parts) != 2:
            raise ValueError(f"无效关键词行: {line}")
        case_number, raw_groups = parts
        groups: list[list[str]] = []
        for raw_group in raw_groups.split("||"):
            terms = [term.strip() for term in raw_group.split("|") if term.strip()]
            if terms:
                groups.append(terms)
        result[f"graham-{int(case_number):03d}"] = groups
    return result


def load_cases() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    rubric = parse_rubric()
    cases = load_cases()
    case_ids = {str(case["id"]) for case in cases}
    positive_case_ids = {
        str(case["id"])
        for case in cases
        if not bool(case.get("expect_abstention", False))
    }
    missing = positive_case_ids - set(rubric)
    extra = set(rubric) - case_ids
    if missing or extra:
        raise ValueError(f"题号不一致: missing={sorted(missing)}, extra={sorted(extra)}")

    output_cases: dict[str, dict[str, object]] = {}
    for case in cases:
        case_id = str(case["id"])
        abstention = bool(case.get("expect_abstention", False))
        groups = [] if abstention else rubric[case_id]
        entry: dict[str, object] = {
            "expect_abstention": abstention,
            "required_groups": groups,
            # A previous import had replacement characters in some legacy
            # expected_keywords fields.  Never copy those into this audited
            # sidecar; required_groups above is the source of truth.
            "supporting_keywords": [
                str(value)
                for value in case.get("expected_keywords", [])
                if "?" not in str(value)
            ],
        }
        if case_id == "graham-084":
            entry["annotation_note"] = (
                "该题是电商运营问题，与 Paul Graham 知识库无关；保留用于暴露题集标注异常。"
            )
        output_cases[case_id] = entry

    payload = {
        "dataset": DATASET_PATH.name,
        "version": "2026-08-01-keyword-rubric-v2",
        "description": (
            "答案关键字审阅标注：required_groups 是必须覆盖的核心概念组，"
            "组内支持中英文同义表达；supporting_keywords 仅作辅助审阅。"
            "本文件只用于评测，不进入 data、Chroma 或 Chat Prompt。"
        ),
        "cases": output_cases,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"写入 {OUTPUT_PATH}：{len(output_cases)} 题，{sum(bool(v['required_groups']) for v in output_cases.values())} 题有必需关键词组")


if __name__ == "__main__":
    main()
