# 每日更新维护 / Daily maintenance

本文件是现有 **Update Biomedical AI Lab** ChatGPT 定时任务的执行入口。计划为每日 **Asia/Shanghai 09:00**。维护目标是持续发现与核实模型，完成能被真实证据支持的更新；当天没有合格新增时，保留有证据的 `no change` 结论。不得为制造更新而换名、降低实现标准或添加无关实验路径。

## 1. 先定位实际版本与预算

读取当前仓库、`teaching/README.md`、`catalog.json`、`sources.json`、`manifest.json` 和 `maintenance/status.json`。确认 git 分支、HEAD、工作区改动及现有权限。稳定模型 ID 和原有课程正文必须保留；实时目录可继续追加，137 模型完整专著是有日期的固定版本，不能静默改写。

发现窗口从 `maintenance/status.json` 中的 `last_successful_scan_utc` 向前重叠 7 天开始，到本次实际检查时间结束；尚无成功记录时，才从本次检查时间向前取 30 天。记录实际 UTC 起止时间，不能用上次尝试时间或内容变更时间替代成功扫描时间，也不能因 `partial` 或 `failed` 而跳过待补查区间。

每次最多深入核验 3 个候选，最多新增或升级 2 项。使用官方论文、作者仓库、发行说明、官方模型卡；区分首次公开、论文上线、仓库提交和权重版本日期。一般更新不安装新的机器学习依赖、不下载模型权重、不运行长训练。需要执行适配器时在已有隔离环境按每项 120 秒 / 3 GiB 的软预算验证，超限如实记录。

## 2. 执行可记录的发现与核查

在仓库根目录可用标准库运行。以下 `--days 30` 是尚无成功记录时的首次扫描示例，也可作为完整发现窗口内最近 30 天的有界子扫描：

```bash
python teaching/tools/maintenance.py scan --days 30 --max-items 20 --timeout 10 --output /tmp/biomed-discovery.json
```

后续运行应按第 1 节计算窗口，不能固定回看 30 天。`scan --days` 只支持最近 1–30 天；所需窗口在此范围内时选择足以覆盖起点的天数，并保留命令实际覆盖日期。若积压窗口超过 30 天，使用带明确日期的一手检索或已连接来源分段补齐更早区间，分别记录查询及实际起止日期；单次 `--days 30` 不能代表整个积压窗口已检查。预算内未补齐时记录 `partial`，不推进 `last_successful_scan_utc`。

该命令访问一个 arXiv 最新条目查询及 MONAI、MedSAM、SAM2 三个明确列出的 release feed；不代表覆盖整个学科。最多 4 个请求，每请求至多 10 秒、2 MiB，无身份凭据、无重定向、无上游代码执行。它输出检索证据，不会自行将候选列为已实现模型，也不会将一次 API 检查写成维护成功。

若运行环境禁止直接联网，可以使用已连接的 GitHub 和网页检索能力读取相同一手来源。保留实际查询、时间、原 URL、固定 commit、读取范围与失败原因；不要仅把工具报错重新标为成功。新候选还须检查论文元数据和方法所必需的输入信息、作者源码是否实际存在、根许可证与权重/数据的独立许可、依赖和低配置运行条件。不可因摘要声称“code will be released”就记录为可运行。

## 3. 决定如何纳入

优先复用可合法使用的作者实现，其次是有方程/结构验收的独立实现。只有完整真实路径通过当前检查，才更新适配器状态；尚未完成时使用 `source_route`。新模型必须有唯一递增编号、稳定 ID、一手来源及许可证说明、Atlas 详情页、课程、独立轻量 SQLite 数据库和明确的实现缺口。数据库是合成教学材料，不得冒充真实 CT、X 光、患者数据或原模型输出。

修改以下数据应一起核对：`biomed-ai-atlas/catalog.json`、`biomed-ai-atlas/models/<stable-id>.json`、`teaching/catalog.json`、`teaching/sources.json`、`teaching/data/manifest.json`。每个 family 的 count 要与目录一致；课程数量从目录生成，不使用固定 137 gate。保留所有旧 ID。

### 离线构建与历史证据

```bash
python teaching/tools/maintenance.py validate
python teaching/tools/build_site.py
```

网页构建仅使用标准库，读取绑定实际可执行源码树摘要的 `adapter_snapshot.json`。新增来源档案或 `source_route` 不要求安装 PyTorch；如果实际模型代码、数据生成代码、运行器、第三方源码或依赖清单改变，快照不匹配会拒绝离线复用，不能悄悄漏掉模型。

确实修改运行代码后，在固定版本的实验环境执行：

```bash
python teaching/tools/build_site.py --refresh-adapters
python -m biomed_lab verify MODEL_NUMBER --steps 20 --output runs/checked-model.json
```

后一个命令从 `teaching/` 目录运行。本每日任务以真实 CPU 执行 20 步作为默认完整验证要求，须显式传入 `--steps 20`；当前 `verify` CLI 省略该参数时默认仅 2 步，`--steps 2` 只能用于快速诊断，不得算作本任务的完整验证。未完成 20 步或超出预算时按实际结果记录，不降低步数后宣称通过。

`--refresh-adapters` 只读取构造器，不等于训练验证。合并新证据时按 model_num 保留未重跑条目的原始记录，不覆盖 `results/verification.json` 为仅单一模型。新记录必须来自实际运行，绝不改写旧记录的时间、hash、步数或状态去匹配当前版本。

如果只是来源档案改变，而运行代码与对应数据仍与已归档版本相同，生成页将旧成功记录显示为 `historical`，保留原 run_id、环境和 source_code_sha256；不写成当天重跑。`verified` 仅计完整摘要仍匹配的记录，`historical_verified` 单列已有版本的成功记录；两者都不是原论文结果复现。

## 4. 验证、生成并记录真实结果

```bash
python -m unittest discover -s teaching/tests -p test_maintenance.py -v
python teaching/tools/maintenance.py validate
python teaching/tools/audit_sources.py
python teaching/tools/build_site.py
python3 .github/scripts/prepare_site.py
python3 .github/scripts/check_site.py
node .github/scripts/test_security.cjs
```

以上命令从仓库根目录按顺序运行，保持 **build → prepare → check → security tests**。新生成或修改的 HTML 必须经过 `prepare_site.py` 的安全处理和 CSP 更新，再通过静态与安全检查，避免内联事件等内容破坏 CSP。若检查后重新构建或修改网页，须重新执行 prepare 及其后的检查，不能直接发布未经处理的产物。

检查本次真实 diff、生成后的 manifest/课程、链接、模型编号与数据库统计。对网页变动执行仓库现有静态与浏览器检查；对本次模型代码执行对应结构/损失/梯度测试。不要把未运行的机器学习测试写为通过。

把完整证据存为临时 JSON，再由下列命令验证并生成公开状态：

```bash
python teaching/tools/maintenance.py record --input /tmp/biomed-reviewed-run.json
```

记录结构参照 `maintenance/runs/` 的实际记录，必须包括：`schema_version`、唯一 `run_id`、`status`、UTC 开始/完成时间、日期窗口、实际 `searches`、逐条 `source_checks`、候选及纳入理由、`changes`、真实 `validation`、中英结果摘要。`changes` 用稳定 ID 数组 `added_model_ids` / `upgraded_model_ids`；维护工具和 UI 的改动不计作新增模型。

状态为 `success`、`partial`、`failed`。未解决的必需来源失败或检索失败不能标成功。内容已核实并纳入但其他候选来源不可访问，可以记录 `partial` 并逐项说明。没有新模型且检索完成仍可标成功。`last_successful_scan_utc` 只在完整成功时前进；`last_content_change_utc` 只在实质课程/模型变化时前进；失败记录不会清空上次成功时间。

完整记录写入不可覆盖的 `maintenance/runs/<run_id>.json`，轻量历史摘要为 `maintenance/history.json`，网页读取 `maintenance/status.json`。任何未核对的临时故障都应保留；同一 run_id 的同内容重放幂等，不允许替换原证据。

## 5. 发布与验收

当前自动任务在 ChatGPT 中运行，**没有另建重复的 GitHub 定时任务**。用户已授权这一维护范围内的核实、实现、验证和更新发布；具体权限仍以当前连接为准。

确认本地检查通过后，使用已有 GitHub 连接将已审阅文件提交到正确仓库分支；更新 ref 时使用读取到的 `expected_sha` 和非 force 更新。若 HEAD 改变，重新读取并合并，不能覆盖其他人修改。连接提交正常触发当前 Pages build 和质量检查；检查实际 action 结果及最终页面，不能只因 commit 成功就宣称已发布。不外发消息、不创建重复任务、不安装付费边缘服务。

仅当未来采用 GitHub Actions 自身的定时写入时，才需另外检查 `GITHUB_TOKEN` 递归触发限制；那不是当前 ChatGPT 连接提交的执行路径。某一步因权限或自动审批受阻时，保留已完成成果，明确受阻的动作与原因，不谎称上线。

## 每日交付标准

给出当次日期、检索范围、新增/升级/延期项、实际验证范围、记录链接、commit 与 Pages 状态。无变化、部分失败和预算停止都是可接受的真实结论；“源文件存在”“构建命令成功”“已有 102 项旧证据”不能替代本次模型执行或线上验收。
