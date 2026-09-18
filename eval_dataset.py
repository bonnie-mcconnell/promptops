EVAL_PROMPTS = [
    # Creative - vague vs specific
    {"prompt": "write about dogs", "goal": "engaging blog intro"},
    {"prompt": "tell me about the ocean", "goal": "make it interesting"},
    {"prompt": "write a short story opening about a lighthouse keeper", "goal": "atmospheric, sets a mysterious tone"},
    {"prompt": "write a poem about autumn", "goal": "melancholic but hopeful"},
    {"prompt": "describe a busy city street", "goal": "vivid, sensory details"},
    {"prompt": "write the opening line of a mystery novel", "goal": "hooks the reader immediately"},
    {"prompt": "write about a character discovering a hidden room", "goal": "builds suspense"},
    {"prompt": "write a scene set on a spaceship", "goal": "tense, claustrophobic mood"},
    {"prompt": "describe a childhood memory of summer", "goal": "nostalgic, warm tone"},

    # Informational/explanatory
    {"prompt": "explain photosynthesis", "goal": "clear for a 10-year-old"},
    {"prompt": "explain how interest rates affect the housing market", "goal": "for someone with no finance background"},
    {"prompt": "what is dependency injection", "goal": "explain to a junior developer"},
    {"prompt": "explain how vaccines work", "goal": "accessible to a skeptical adult reader"},
    {"prompt": "explain the difference between weather and climate", "goal": "clear for a high school student"},
    {"prompt": "explain what a black hole is", "goal": "avoid jargon, keep it intuitive"},
    {"prompt": "explain how blockchain works", "goal": "for someone who's only used a bank app before"},
    {"prompt": "explain why the sky is blue", "goal": "simple enough for a curious child"},
    {"prompt": "explain what inflation means", "goal": "practical, relatable to everyday spending"},

    # Business/professional
    {"prompt": "describe our Q3 results", "goal": "concise executive summary, under 100 words"},
    {"prompt": "write a status update for my team", "goal": "professional, highlights blockers"},
    {"prompt": "draft a cover letter opening for a data analyst role at a fintech startup, emphasizing my SQL and Python skills", "goal": "professional but not stiff"},
    {"prompt": "write a performance review summary for a team member", "goal": "constructive, balanced, specific"},
    {"prompt": "draft an email declining a meeting invite", "goal": "polite but firm"},
    {"prompt": "write a project kickoff announcement", "goal": "clear on scope and next steps"},
    {"prompt": "draft a follow-up email after a job interview", "goal": "professional, expresses genuine interest"},
    {"prompt": "write a LinkedIn post about starting a new job", "goal": "authentic, not overly self-promotional"},
    {"prompt": "draft an out-of-office auto-reply", "goal": "clear on when I'll respond, who to contact instead"},

    # Technical
    {"prompt": "write a function docstring", "goal": "technical, precise, follows Google style"},
    {"prompt": "explain this error: NullPointerException", "goal": "help a beginner debug it"},
    {"prompt": "describe our API's rate limiting", "goal": "clear for external developers reading our docs"},
    {"prompt": "write a commit message for a bug fix in the login flow", "goal": "concise, follows conventional commits style"},
    {"prompt": "explain what a race condition is", "goal": "clear for someone new to concurrent programming"},
    {"prompt": "write a code review comment about a missing null check", "goal": "constructive, not condescending"},
    {"prompt": "explain the difference between SQL and NoSQL databases", "goal": "practical, for a junior engineer choosing between them"},
    {"prompt": "describe how our caching layer works", "goal": "clear for a new engineer onboarding"},
    {"prompt": "write a README section explaining installation steps", "goal": "step-by-step, assumes no prior setup"},

    # Persuasive
    {"prompt": "pitch a productivity app", "goal": "persuasive, targeting busy professionals"},
    {"prompt": "write ad copy for a coffee subscription", "goal": "punchy, under 30 words"},
    {"prompt": "write a fundraising appeal for a local animal shelter", "goal": "emotionally resonant, includes a clear call to action"},
    {"prompt": "pitch a new feature idea to my manager", "goal": "persuasive, backed by a clear rationale"},
    {"prompt": "write a tagline for a sustainable clothing brand", "goal": "memorable, communicates the brand's values"},
    {"prompt": "write a pitch for a podcast idea", "goal": "convince a producer it's worth greenlighting"},
    {"prompt": "write a tweet promoting a webinar", "goal": "persuasive, under 280 characters"},
    {"prompt": "pitch a gym membership to someone who hates gyms", "goal": "address objections directly"},
    {"prompt": "write a landing page headline for a budgeting app", "goal": "clear value proposition, benefit-driven"},

    # Deliberately ambiguous / minimal goal
    {"prompt": "summarize this meeting", "goal": "make it useful"},
    {"prompt": "write something about climate change", "goal": "informative"},
    {"prompt": "help me write an email", "goal": "make it good"},
    {"prompt": "write a description for this product", "goal": "sell it"},
    {"prompt": "give me feedback on this idea", "goal": "be helpful"},
    {"prompt": "write something for social media", "goal": "get engagement"},
    {"prompt": "help me plan a presentation", "goal": "make it work"},
    {"prompt": "write a message to my boss", "goal": "handle it well"},

    # Instructional
    {"prompt": "explain how to set up a git repository", "goal": "step-by-step for a beginner"},
    {"prompt": "explain how to make a basic omelette", "goal": "clear steps for someone who's never cooked eggs"},
    {"prompt": "explain how to change a flat tire", "goal": "step-by-step, safety-conscious"},
    {"prompt": "explain how to set up a budget spreadsheet", "goal": "beginner-friendly, no finance background assumed"},
    {"prompt": "explain how to write a resignation letter", "goal": "step-by-step, professional tone"},
    {"prompt": "explain how to do a basic SWOT analysis", "goal": "clear for someone who's never done one before"},
    {"prompt": "explain how to back up a laptop", "goal": "simple steps for a non-technical person"},
    {"prompt": "explain how to negotiate a salary offer", "goal": "practical, step-by-step for a first-timer"},
]