"""
===============================================================
Script Name : Custome_Vocab.py
Project Name: Assessment 1: NLP Recommendation Engine
Unit Name   : MA5851
Description : Custom Vocabulary for the NLP Recommendation Engine
Author      : Alexander Floyd
Student ID  : JC502993
Email       : alex.floyd@my.jcu.edu.au
Date        : December 05, 2024
Version     : 1.0
===============================================================
"""

science_vocab = [
    "biology", "chemistry", "physics", "atom", "energy", "photosynthesis", "quantum mechanics", "ecosystem", "genetics", 
    "evolution", "chemical reactions", "molecules", "elements", "compounds", "acids", "bases", "atomic structure", "periodic table", 
    "stoichiometry", "chemical bonding", "optics", "electricity", "magnetism", "thermodynamics", "force", "motion", "gravity", 
    "planetary systems", "natural selection", "gene expression", "ecosystem services", "climate change", "biodiversity", "pollution", 
    "solar energy", "climate adaptation", "geology", "earth systems", "earth's layers", "human biology", "plant biology", "animal behavior", 
    "earthquake", "volcanoes", "weather", "acids and bases", "photosynthetic organisms", "global warming", "environmental chemistry", 
    "marine biology", "climate models", "space exploration", "human anatomy", "greenhouse gases", "renewable energy", "health science"
]


mathematics_vocab = [
    "algebra", "geometry", "calculus", "probability", "equations", "functions", "trigonometry", "derivatives", "integrals", 
    "limits", "differentiation", "continuity", "optimization", "infinitesimals", "mean", "median", "mode", "variance", 
    "standard deviation", "sampling", "distribution", "sine", "cosine", "tangent", "Pythagorean theorem", "right triangles", 
    "angles", "unit circle", "vectors", "linear algebra", "polynomials", "quadratic equations", "inequalities", "logarithms", 
    "exponents", "probability theory", "combinatorics", "statistics", "factorization", "set theory", "graph theory", "permutations", 
    "combinations", "calculus limits", "trigonometric identities", "mean square deviation", "polynomial equations", "math proofs", 
    "conic sections", "mathematical modeling", "mathematical functions"
]

english_vocab = [
    "literacy", "reading", "writing", "comprehension", "grammar", "spelling", "punctuation", "sentence structure", 
    "paragraph structure", "essay writing", "creative writing", "descriptive writing", "narrative writing", "persuasive writing", 
    "expository writing", "argumentative writing", "vocabulary", "synonyms", "antonyms", "literary devices", "syntax", "morphology", 
    "phonics", "phonemic awareness", "literary analysis", "textual analysis", "literary genres", "fiction", "non-fiction", 
    "poetry", "drama", "novels", "short stories", "biography", "autobiography", "memoir", "dialogue", "figurative language", 
    "metaphor", "simile", "alliteration", "personification", "tone", "mood", "theme", "symbolism", "rhyme", "rhetoric", "sentence types", 
    "active voice", "passive voice", "tense", "parts of speech", "adjectives", "verbs", "nouns", "pronouns", "adverbs", "prepositions", 
    "writing conventions", "text features", "structure", "organization", "academic writing", "proofreading", "editing", "summarization", 
    "fluency", "writing mechanics", "speech"
]


languages_vocab = [
     "morphology", "phonetics", "semantics", "sentence structure", "verb conjugation", "tense", "pronunciation", 
    "vocabulary acquisition", "language proficiency", "second language learning", "language immersion", "bilingualism", "fluency", 
    "reading comprehension", "writing skills", "speaking skills", "listening skills", "language arts", "language teaching", 
    "language acquisition", "multilingualism", "linguistic diversity", "phonology", "language family", "literacy", "dialects", 
    "language variation", "code-switching", "language skills", "semantic analysis", "pragmatics", "communication skills", 
    "language structure", "word formation", "sociolinguistics", "linguistic anthropology", "language learning strategies", "language acquisition theory", 
    "cultural context", "foreign language teaching", "second language acquisition", "linguistic competence", "literary criticism", 
    "translation studies", "reading fluency", "cognitive linguistics", "Mandarin", "Spanish", "French", "German", "Italian", "Japanese", 
    "Indonesian", "Arabic", "Korean", "Vietnamese", "Russian", "Latin", "ancient Greek", "foreign language exams", "cross-cultural communication"
]


humanities_vocab = [
    "philosophy", "ethics", "logic", "metaphysics", "epistemology", "sociology", "psychology", "social studies", "history", 
    "human behavior", "cultural studies", "religion", "political science", "law", "economics", "psychodynamic theory", "social structure", 
    "human development", "consciousness", "neuroscience", "human rights", "culture", "deviance", "socialization", "gender studies", 
    "class systems", "inequality", "race and ethnicity", "moral philosophy", "political ideologies", "democracy", "authoritarianism", 
    "feminism", "post-colonialism", "globalization", "global warming", "colonialism", "indigenous rights", "cultural anthropology", 
    "ethnography", "archaeology", "evolution", "human societies", "kinship", "rituals", "anthropological theory", "democracy", 
    "authoritarianism", "political ideologies", "governance", "elections", "international relations", "policy analysis", "political theory", 
    "religious studies", "spirituality", "ethnic studies", "international relations", "peace studies", "public policy", "global conflict", 
    "social change", "citizenship", "identity", "history of ideas"
]


environmental_science_vocab = [
    "ecosystem", "biodiversity", "sustainability", "pollution", "climate change", "greenhouse gases", "renewable energy", 
    "carbon footprint", "energy conservation", "resource management", "climate action", "environmental ethics", "waste management", 
    "conservation", "ecology", "water quality", "deforestation", "land degradation", "recycling", "soil erosion", "ecosystem services", 
    "renewable resources", "solar power", "wind energy", "sustainable agriculture", "carbon offset", "environmental legislation", 
    "environmental science", "global warming", "acid rain", "water conservation", "habitat loss", "species extinction", "pollution control", 
    "reforestation", "carbon sequestration", "plastic waste", "ecosystem restoration", "environmental education", "green infrastructure", 
    "alternative energy", "organic farming", "sustainable practices", "water management", "eco-friendly", "recyclable materials", 
    "climate adaptation", "earth systems", "marine biology", "landfills", "environmental protection", "conservation efforts"
]


design_and_technology_vocab = [
    "design thinking", "innovation", "user-centered design", "prototyping", "3D modeling", "CAD", "electronics", "mechanical systems", 
    "robotics", "automation", "manufacturing", "engineering", "materials science", "industrial design", "architecture", "sustainable design", 
    "aesthetics", "functionality", "user interface", "product development", "technology integration", "control systems", "programming", 
    "software development", "hardware design", "system architecture", "product testing", "quality assurance", "rapid prototyping", 
    "manufacturing techniques", "CNC machining", "laser cutting", "material handling", "energy-efficient design", "cost estimation", 
    "construction design", "building codes", "product design", "tool design", "environmental considerations", "sustainability", 
    "environmental impact", "renewable resources", "green technology", "life cycle analysis", "system integration", "user experience", 
    "construction methods", "team collaboration", "technology management", "material properties", "safety standards", "project management"
]


business_and_law_vocab = [
    "entrepreneurship", "marketing", "finance", "business strategy", "corporate governance", "management", "leadership", 
    "business ethics", "accounting", "financial management", "business operations", "consumer behavior", "global trade", 
    "international business", "corporate law", "contract law", "civil law", "criminal law", "constitutional law", "litigation", 
    "lawsuit", "torts", "intellectual property", "legal studies", "political economy", "economic theory", "market research", 
    "supply and demand", "economics", "banking", "trade agreements", "negotiation", "consumer protection", "investment", 
    "corporate responsibility", "globalization", "global finance", "financial markets", "trade law", "taxation", "public policy", 
    "labor laws", "competition law", "social responsibility", "management theories", "cost-benefit analysis", "leadership theory", 
    "financial planning", "business planning", "marketing strategies", "business analysis", "business ethics", "corporate finance"
]


practical_studies_vocab = [
    "vocational education", "trade skills", "apprenticeships", "certification", "workplace skills", "construction", 
    "plumbing", "electrical", "mechanical", "food technology", "home economics", "work studies", "PDHPE", "health and safety", 
    "time management", "teamwork", "problem-solving", "project management", "customer service", "communication skills", 
    "workplace safety", "risk management", "first aid", "personal protective equipment", "power tools", "machinery", 
    "technical drawing", "CAD software", "3D printing", "fabrication", "measurement techniques", "material handling", 
    "sustainable practices", "energy conservation", "welding", "tool maintenance", "metalworking", "manufacturing", 
    "woodworking", "construction planning", "engine components", "sewing", "knitting", "baking", "cooking", "nutrition", 
    "personal finance", "electrical circuits", "problem-solving skills", "construction design", "communication in the workplace", 
    "engineering", "auto mechanics", "landscaping", "customer relations"
]


performance_arts_vocab = [
    "performing arts", "theatre", "choreography", "dance", "music", "drama", "acting", "performance", "script", "audition", 
    "stage design", "set design", "lighting design", "costume design", "sound design", "rehearsals", "improvisation", 
    "directing", "acting techniques", "performance skills", "stage presence", "stage directions", "movement", 
    "body language", "rhythm", "melody", "harmony", "orchestra", "choir", "instrumentation", "ensemble", "vocal performance", 
    "musical theatre", "dance techniques", "hip-hop", "ballet", "contemporary dance", "classical music", "musical composition", 
    "theatrical effects", "visual arts", "media arts", "media production", "sound effects", "stage management", "theatrical lighting", 
    "props", "performance analysis", "audience engagement", "creative expression", "non-verbal communication", "movement vocabulary"
]
