# Workspace Rules / 工作区规则

This file defines soft rules for Codex when working in this workspace.
本文件用于为 Codex 在当前工作区中的行为提供软规则约束。

## 1. Scope / 作用范围

- Only operate within the currently opened workspace unless the user explicitly asks otherwise.
- 仅在当前打开的工作区内执行操作，除非用户明确提出其他要求。

- Treat the workspace root as the default boundary for reading, writing, moving, renaming, and deleting files.
- 将工作区根目录视为读取、写入、移动、重命名和删除文件的默认边界。

## 2. Path Rules / 路径规则

- Prefer relative paths over absolute paths whenever possible.
- 在可能的情况下，优先使用相对路径，而不是绝对路径。

- Do not proactively use absolute Windows paths such as `C:\...` for file operations unless the user explicitly requests it.
- 除非用户明确要求，否则不要主动使用 `C:\...` 这类 Windows 绝对路径执行文件操作。

- If an absolute path points outside the current workspace, stop and ask for confirmation before using it.
- 如果绝对路径指向当前工作区之外，先停止操作并请求用户确认。

## 3. File Access / 文件访问

- Prefer opened files, selected text, and files already mentioned in the conversation as primary context.
- 优先将已打开文件、已选中文本和对话中已经提到的文件作为主要上下文。

- Do not scan unrelated folders outside the workspace just because they exist on disk.
- 不要仅因为磁盘上存在，就主动扫描工作区外无关的目录。

- If a task appears to require files outside the workspace, explain why and ask first.
- 如果任务看起来需要访问工作区外的文件，请先说明原因并征求确认。

## 4. Edit Safety / 编辑安全

- Only create, edit, move, or delete files inside the current workspace unless the user explicitly approves a wider scope.
- 仅在当前工作区内创建、编辑、移动或删除文件，除非用户明确批准扩大范围。

- Never modify system folders, user home configuration, or other repositories unless the user clearly asks for it.
- 不要修改系统目录、用户主目录配置或其他仓库，除非用户明确提出要求。

- When in doubt about scope, choose the safer option and ask before proceeding.
- 如果对操作范围有疑问，优先选择更保守的做法，并在继续之前先确认。

## 5. Communication / 沟通方式

- Be explicit when a requested action stays inside the workspace.
- 当某个请求完全限定在工作区内时，请明确说明这一点。

- Be explicit when an action would cross the workspace boundary.
- 当某个操作将跨出工作区边界时，也请明确说明。

- If the user asks for a wider-scope action, confirm the consequence before proceeding.
- 如果用户要求更大范围的操作，请先确认其影响，再继续执行。

## 6. External Blockers / 外部阻塞因素

- If progress is blocked by an external factor such as package download failure, network/proxy issues, missing system dependencies, account permissions, or unavailable services, stop and report the blocker clearly instead of writing large amounts of workaround code that changes the project direction.
- 如果进度被外部因素阻塞，例如拉包失败、网络或代理问题、系统依赖缺失、权限不足或外部服务不可用，应先暂停并清楚说明阻塞原因，而不是为了绕开问题编写大量会改变项目方向的替代代码。

- In these cases, prefer asking the user to handle the external dependency manually or confirm the next step before adding compatibility layers, fallback implementations, or temporary replacement logic.
- 在这类情况下，优先请用户手动处理外部依赖，或先确认下一步，再决定是否添加兼容层、回退实现或临时替代逻辑。
