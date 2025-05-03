SENTENCES_LIST = {
    "phrases_internet": [
        "ratio + L + cope + seethe",  # from poking around the internet
        "Python enumerate() Function",  # w3schools
        "Top 1% Commenter",  # reddit
        "Eggs US – Price – Chart",  # HN, ages ago - from the old benchmark
        "8 FUCKING IRONSHIPS (& gunboat friends)",  # r foxhole
        "Minimum Viable Blog",  # HN frontpage
    ],
    "short_sentence": [
        "This isn’t what I ordered.",  # wrote myself
        "The best translator is a hybrid translator",  # internet
        "Will this be available to download on GitHub?",
        "It doesn't matter now.",  # wrote myself
        "Try refreshing the page.",  # wrote myself
        "This reads like an LLM wrote it.",  # https://news.ycombinator.com/item?id=43870969
    ],
    "medium_sentence": [
        "And a few things I think would be cool, but aren't core to the idea:",  # from r foxhole
        "Hell, just the fact that there's 3 officer uniforms in one picture is funny enough to me.",
        "The risk was calculated, but the variables were bollocks.",  # added swearing to a somewhat common meme-phrase
        "The Nuenki browser extension finds sentences in the websites you visit.",  # nuenki front page!
        "idea: bounce it back and forth with wrapped encryption from a far probe.",  # my own notes
        "At father's command he'd had a crown made, a simple gold circlet.",  # intentionally irritating, literary, overdone sentence from https://www.reddit.com/r/WritingPrompts/comments/1kbhw9k/comment/mpwcpgu/
    ],
    "long_sentence": [
        "The dev feels that there won't really be a point of adding a new vehicle that can only do the same things as the other existing ones.",  # r vtolvr. I felt that it was sufficiently meandering
        "And yet - and yet - that was the very problem, the crux of the issue, the source of all that conflict: Their inability to agree upon anything.",  # wrote myself to be as irritating to translate as possible. The only thing it's missing is nested subordinate clauses
        "While its coherence is slightly lower than its peers (more on that in a moment), it is the most idiomatic model while also being far more consistent, with a much lower standard deviation across all three metrics.",  # nuenki blog post
        "This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples",  # "attention is all you need" :P
        "The objective, of course, was never to solve the actual problem: No, the real reason they had gone through all of that bother was to *appear* to be working on the issue, never mind actually solving it.",  # more meandering text
        "The fact that you got a positive-sloping line out at all from the regression has to do more with the positions of the outliers than anything else.",  # https://news.ycombinator.com/item?id=43870969
    ],
    "paragraph": [
        "I built that, and made it open source. It turns out that you can! While its coherence is slightly lower than its peers (more on that in a moment), it is the most idiomatic model while also being far more consistent, with a much lower standard deviation across all three metrics. It works by taking the top 3-4 models for a given language (based on this research), translating with them, then having a judge model (currently redacted) consider the strengths and weaknesses of each translation and merge them together in an idiomatic way.",  # from https://nuenki.app/blog/the_best_translator_is_a_hybrid_translator, except I replaced "GPT-4.1" with "redacted" because I felt mentioning specific models would do little to help the "model-lineage-bias" issue!
        "Pursuant to Article 8(C) of Directive 121, and in light of the recommendations of the consultion on inter-departmental affairs issued by the Department of Departmental Affairs, the Commission has decided to consult to adopt a preliminary position regarding the regulation of meat. In accordance with Regulation 7(B), the preliminary position will be subject to public consultation pursuant with the Public Consultation Act of 1988.",  # wrote it myself. Loosely inspired by "Yes, Minister"
        "It is in this case that we deploy the Four Stage Strategy. First: Nothing is going to happen. Second: Yes, something may in fact happen, but we shouldn't do anything about it. Third: Yes, something is happening, but there's nothing we *can* do! Third: Alright, maybe there was something we could do, but it's too late now. Oops.",  # Highly inspired by "Yes, Minister"
        "In a quiet forest, where the sun peeked through the trees like golden butter, Ellie the Elephant was not happy. Ellie couldn't find her hat! She asked the squirrels, who were too busy playing tag. She asked the birds, but they were too busy singing. At last, it was Mister Turtle who found it while munching a pile of autumn leaves.",  # my own; attempting to do it more like a child's story
        "The people who get what they want in these situations are the ones who are prepared to behave sufficiently unreasonably, and with sufficient stubbornness. This is a second order consequence of 'unaccountability' that Davies misses. For the customer, or the object of the system, it incentivises unpleasant behaviour - as unpleasant as possible - because it's often the only way to trigger the exception / escalation / special case, and get what you want.",  # comment on HN: https://news.ycombinator.com/item?id=43877301 - with some minor changes
        'They then run stepwise regression to determine variance contributions, seemingly ignoring their earlier results, and this leads to almost no contribution from numeracy. Why? Because they have ~10% shared variance and stepwise regression is greedy - will just take whatever you give it first. I can\'t mention this part enough. If you got a second, very similar language test, and added it to the model, you would _also_ find it has almost no unique variance added. Every thing they measure is incredibly noisy and they do not once attempt to deal with this. Human based reviewers, time-to-completion, etc. p-value for "language learning is more significant than numeracy" on the values they give (Steiger test) gives 0.772. Utterly insignificant.',  # https://news.ycombinator.com/item?id=43870969
    ],
}
