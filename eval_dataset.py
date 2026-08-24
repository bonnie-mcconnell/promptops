EVAL_PROMPTS = [
    # Creative - vague vs specific
    {"prompt": "write about dogs", "goal": "engaging blog intro"},
    {"prompt": "tell me about the ocean", "goal": "make it interesting"},
    {"prompt": "write a short story opening about a lighthouse keeper", "goal": "atmospheric, sets a mysterious tone"},

    # Informational/explanatory
    {"prompt": "explain photosynthesis", "goal": "clear for a 10-year-old"},
    {"prompt": "explain how interest rates affect the housing market", "goal": "for someone with no finance background"},
    {"prompt": "what is dependency injection", "goal": "explain to a junior developer"},

    # Business/professional
    {"prompt": "describe our Q3 results", "goal": "concise executive summary, under 100 words"},
    {"prompt": "write a status update for my team", "goal": "professional, highlights blockers"},
    {"prompt": "draft a cover letter opening for a data analyst role at a fintech startup, emphasizing my SQL and Python skills", "goal": "professional but not stiff"},

    # Technical
    {"prompt": "write a function docstring", "goal": "technical, precise, follows Google style"},
    {"prompt": "explain this error: NullPointerException", "goal": "help a beginner debug it"},
    {"prompt": "describe our API's rate limiting", "goal": "clear for external developers reading our docs"},

    # Persuasive
    {"prompt": "pitch a productivity app", "goal": "persuasive, targeting busy professionals"},
    {"prompt": "write ad copy for a coffee subscription", "goal": "punchy, under 30 words"},

    # Deliberately ambiguous / minimal goal
    {"prompt": "summarize this meeting", "goal": "make it useful"},
    {"prompt": "write something about climate change", "goal": "informative"},

    # Instructional
    {"prompt": "explain how to set up a git repository", "goal": "step-by-step for a beginner"},
]