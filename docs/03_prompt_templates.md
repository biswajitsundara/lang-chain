## Prompt Template in LangChain

* A structured way to create prompts dynamically for LLMs.
* Uses placeholders (variables) that get filled at runtime.
* Helps keep prompts reusable, consistent, and clean.
* Separates prompt logic from hardcoded text.
* Reduces duplication and improves maintainability.

```console
//instead of this
Explain AI to a beginner

//we define a template
Explain {topic} to a {audience}


//Then pass:
topic = "AI"
audience = "beginner"
-> The model receives: "Explain AI to a beginner"
```