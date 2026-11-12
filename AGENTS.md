# AGENTS.md

## Notes on repository structure
- to be documented

## Useful Resources
- to be documented

## General Rules
- Unless specified, make single plot figures. Do not attempt to make larger figures with a lot of text.
- Do not ever save plots as PDF files unless explicitly instructed to.
- Try to be consistent across plots with colors. For example, if you use the hex code FFC0CB to color a superpopulation in one plot, you should use it again in other plots if appropriate.
- Speak concisely and try to empathise what I understand and what is new information and thus needs to be explained. Avoid information overload or being verbose or needlessly technical. If you can explain it in simple terms, do it.
- Keep shared dataframe row Protocols in a lightweight `row_types.py`, keep single-module types local, describe immutable fields with read-only properties, and cast row iterators once at the pandas boundary using accurate field types.
- Make your changes on the current branch in the current folder, i.e. do not work on code changes in a worktree or other location than the one you are running in if not explicitly instructed to do so.
- When writing a class/dataclass/type/protocol definition, add a comment with an example value if ambigious and not totally obvious from the name and type of the property. If an example is not possible (e.g. because it's a complex object) add a very short description. For example:
    ```python
    # BAD
    @dataclass(frozen=True)
    class _TrainingPolicyModule:
        module_name: str
        module_group: LoraModuleGroup


    # GOOD
    @dataclass(frozen=True)
    class _TrainingPolicyModule:
        module_name: str  # e.g. "skip_connection.conv1d"
        module_group: LoraModuleGroup  # ...
    ```
- src/ should never import or use functionality from scripts/

## Git and GitHub
- When creating new branches, use one of: 
    - feat/{name}: for new features
    - fix/{name}: for fixes, correcting bugs
    - docs/{name}: changes to documentation (e.g. code comments, READMEs or docs/)
    - refactor/{name}: code changes the clean up code, refactor or simplify it (no new features)
    - analysis/{name}: a branch which doesn't change any model features, but just does some analysis to check some phenomena (importantly, no src/ code should be changed on an analysis branch)
    - if none fits, suggest a new category in the chat.

## Naming things
- Refer to docs/development/naming_variables_guide.md for instructions on naming variables properly.

Other than that guide, here's some guidance to follow: 

### Naming things (from kern.io)
#### Variables, types, classes, and other data structures
- They should be nouns.
- They should have more specific names when they could be confused with other similar names.
- Single letter variable names are only acceptable for small inline lambda parameters.

#### Functions and methods
- They all start with present imperative verbs.
- They should be shorter and more generic if the more frequently they are used in the codebase.
- If a function primarily computes a value in-memory, begin with get or compute.
- If a function constructs a new object or data structure, begin with build.
- If a function fetches a value from a remote host, begin with fetch or load.
- If a function returns a boolean value, begin with the prefix is, has, or another present indicative verb.

### Project specific jargon
- tbd