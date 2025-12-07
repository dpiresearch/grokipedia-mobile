from manim import *

class FAQScene(Scene):
    def construct(self):
        # Title Screen
        title = Text("DeepSeek MoE - FAQ Deep Dive", font_size=40, color=YELLOW).to_edge(UP)
        self.play(Write(title))
        self.wait(3)
        self.clear()
        self.wait(0.3)

        # === FAQ #1 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #1", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("Where does the forward pass happen in HuggingFace?", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("Forward Pass Flow", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("Custom modeling_deepseek.py with trust_remote_code=True", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        
        detail = Text("Entry point: DeepseekForCausalLM.forward()", font_size=18).next_to(concept, DOWN, buff=0.2)
        self.play(FadeIn(detail))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept), FadeOut(detail))

        # PART C: Code visualization
        code_title = Text("Code Implementation", font_size=24, color=BLUE).next_to(faq_header, DOWN, buff=0.3)
        code_line1 = Text("outputs = self.model(", font_size=14, font="Monospace").next_to(code_title, DOWN, buff=0.2)
        code_line2 = Text("    input_ids=input_ids,", font_size=14, font="Monospace").next_to(code_line1, DOWN, aligned_edge=LEFT)
        code_line3 = Text("    attention_mask=attention_mask,", font_size=14, font="Monospace").next_to(code_line2, DOWN, aligned_edge=LEFT)
        code_line4 = Text("    ...", font_size=14, font="Monospace").next_to(code_line3, DOWN, aligned_edge=LEFT)
        code_line5 = Text(")", font_size=14, font="Monospace").next_to(code_line4, DOWN, aligned_edge=LEFT)
        
        code_group = VGroup(code_line1, code_line2, code_line3, code_line4, code_line5)
        self.play(FadeIn(code_title))
        self.play(FadeIn(code_line1))
        self.wait(0.5)
        self.play(FadeIn(code_line2))
        self.wait(0.5)
        self.play(FadeIn(code_line3))
        self.wait(0.5)
        self.play(FadeIn(code_line4))
        self.wait(0.5)
        self.play(FadeIn(code_line5))
        self.wait(2)
        self.play(FadeOut(code_title), FadeOut(code_group))

        # PART D: Diagram
        diagram_title = Text("Transformer Flow", font_size=24, color=RED).next_to(faq_header, DOWN, buff=0.3)
        self.play(FadeIn(diagram_title))
        
        # Create boxes
        box1 = Rectangle(width=3, height=0.8, color=BLUE).shift(UP*2)
        label1 = Text("Input IDs", font_size=16).move_to(box1)
        
        box2 = Rectangle(width=3, height=0.8, color=GREEN).next_to(box1, DOWN, buff=1)
        label2 = Text("Embedding", font_size=16).move_to(box2)
        
        box3 = Rectangle(width=3, height=0.8, color=ORANGE).next_to(box2, DOWN, buff=1)
        label3 = Text("Transformer Layers", font_size=16).move_to(box3)
        
        box4 = Rectangle(width=3, height=0.8, color=PURPLE).next_to(box3, DOWN, buff=1)
        label4 = Text("LM Head", font_size=16).move_to(box4)
        
        # Arrows
        arrow1 = Arrow(box1.get_bottom(), box2.get_top())
        arrow2 = Arrow(box2.get_bottom(), box3.get_top())
        arrow3 = Arrow(box3.get_bottom(), box4.get_top())
        
        self.play(Create(box1), Write(label1))
        self.play(Create(arrow1))
        self.play(Create(box2), Write(label2))
        self.play(Create(arrow2))
        self.play(Create(box3), Write(label3))
        self.play(Create(arrow3))
        self.play(Create(box4), Write(label4))
        self.wait(2)
        self.play(FadeOut(diagram_title), FadeOut(VGroup(box1, label1, arrow1, box2, label2, arrow2, box3, label3, arrow3, box4, label4)))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: Custom forward() in modeling_deepseek.py", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # === FAQ #2 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #2", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("What is the max context length for this model?", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("Context Length Limit", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("Defined by max_position_embeddings in config.json", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept))

        # PART C: Code visualization
        code_title = Text("Config Value", font_size=24, color=BLUE).next_to(faq_header, DOWN, buff=0.3)
        code_line = Text('"max_position_embeddings": 4096,', font_size=16, font="Monospace").next_to(code_title, DOWN, buff=0.3)
        self.play(FadeIn(code_title), FadeIn(code_line))
        self.wait(3)
        self.play(FadeOut(code_title), FadeOut(code_line))

        # PART E: Math formula
        math_title = Text("Token Limit", font_size=24, color=PURPLE).next_to(faq_header, DOWN, buff=0.3)
        formula = MathTex(r"\text{Max Tokens} = 4096", font_size=28).next_to(math_title, DOWN)
        self.play(FadeIn(math_title), Write(formula))
        self.wait(2)
        self.play(FadeOut(math_title), FadeOut(formula))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: Max context = 4096 tokens", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # === FAQ #3 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #3", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("How can model handle >4096 tokens without error?", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("RoPE Dynamic Extension", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("RoPE recomputes cache for longer sequences", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept))

        # PART C: Code visualization
        code_title = Text("RoPE Implementation", font_size=24, color=BLUE).next_to(faq_header, DOWN, buff=0.3)
        code_line1 = Text("if seq_len > self.max_seq_len_cached:", font_size=14, font="Monospace").next_to(code_title, DOWN, buff=0.2)
        code_line2 = Text("    self._set_cos_sin_cache(seq_len=seq_len, ...)", font_size=14, font="Monospace").next_to(code_line1, DOWN, aligned_edge=LEFT)
        code_group = VGroup(code_line1, code_line2)
        self.play(FadeIn(code_title))
        self.play(FadeIn(code_line1))
        self.wait(1)
        self.play(FadeIn(code_line2))
        self.wait(2)
        self.play(FadeOut(code_title), FadeOut(code_group))

        # PART D: Diagram
        diagram_title = Text("RoPE Cache Extension", font_size=24, color=RED).next_to(faq_header, DOWN, buff=0.3)
        self.play(FadeIn(diagram_title))
        
        # Boxes
        box1 = Rectangle(width=2.5, height=0.7, color=BLUE).shift(LEFT*3)
        label1 = Text("Seq Len", font_size=16).move_to(box1)
        
        box2 = Rectangle(width=2.5, height=0.7, color=GREEN).shift(ORIGIN)
        label2 = Text("Cache Check", font_size=16).move_to(box2)
        
        box3 = Rectangle(width=2.5, height=0.7, color=ORANGE).shift(RIGHT*3)
        label3 = Text("Recompute", font_size=16).move_to(box3)
        
        # Arrows
        arrow1 = Arrow(box1.get_right(), box2.get_left())
        arrow2 = Arrow(box2.get_right(), box3.get_left())
        
        self.play(Create(box1), Write(label1))
        self.play(Create(arrow1))
        self.play(Create(box2), Write(label2))
        self.play(Create(arrow2))
        self.play(Create(box3), Write(label3))
        self.wait(2)
        self.play(FadeOut(diagram_title), FadeOut(VGroup(box1, label1, arrow1, box2, label2, arrow2, box3, label3)))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: RoPE extends dynamically but quality degrades", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # === FAQ #4 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #4", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("What are embedding layer input/output shapes?", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("Embedding Shapes", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("Token embedding independent of position embeddings", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept))

        # PART C: Code visualization
        code_title = Text("Embedding Layer", font_size=24, color=BLUE).next_to(faq_header, DOWN, buff=0.3)
        code_line1 = Text("self.embed_tokens = nn.Embedding(", font_size=14, font="Monospace").next_to(code_title, DOWN, buff=0.2)
        code_line2 = Text("    config.vocab_size, config.hidden_size", font_size=14, font="Monospace").next_to(code_line1, DOWN, aligned_edge=LEFT)
        code_line3 = Text(")", font_size=14, font="Monospace").next_to(code_line2, DOWN, aligned_edge=LEFT)
        code_group = VGroup(code_line1, code_line2, code_line3)
        self.play(FadeIn(code_title))
        self.play(FadeIn(code_line1))
        self.wait(0.5)
        self.play(FadeIn(code_line2))
        self.wait(0.5)
        self.play(FadeIn(code_line3))
        self.wait(2)
        self.play(FadeOut(code_title), FadeOut(code_group))

        # PART D: Diagram
        diagram_title = Text("Shape Comparison", font_size=24, color=RED).next_to(faq_header, DOWN, buff=0.3)
        self.play(FadeIn(diagram_title))
        
        # Text elements
        text1 = Text("< 4096 tokens:", font_size=20).shift(UP*1.5 + LEFT*3)
        shape1 = Text("[batch, seq<4096, 2048]", font_size=18).next_to(text1, DOWN, buff=0.2)
        
        text2 = Text("> 4096 tokens:", font_size=20).shift(UP*1.5 + RIGHT*3)
        shape2 = Text("[batch, seq>4096, 2048]", font_size=18).next_to(text2, DOWN, buff=0.2)
        
        self.play(FadeIn(text1), FadeIn(shape1))
        self.play(FadeIn(text2), FadeIn(shape2))
        self.wait(2)
        self.play(FadeOut(diagram_title), FadeOut(VGroup(text1, shape1, text2, shape2)))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: Embedding shape = [batch, seq_len, 2048]", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # === FAQ #5 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #5", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("What are attention layer input/output shapes?", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("Attention Shapes", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("Multi-head attention with hidden_size=2048, heads=16", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept))

        # PART E: Math formula
        math_title = Text("Attention Dimensions", font_size=24, color=PURPLE).next_to(faq_header, DOWN, buff=0.3)
        formula1 = MathTex(r"\text{Input: } [B, S, 2048]", font_size=24).next_to(math_title, DOWN)
        formula2 = MathTex(r"\text{Q,K,V: } [B, 16, S, 128]", font_size=24).next_to(formula1, DOWN)
        formula3 = MathTex(r"\text{Output: } [B, S, 2048]", font_size=24).next_to(formula2, DOWN)
        formulas = VGroup(formula1, formula2, formula3)
        self.play(FadeIn(math_title))
        self.play(Write(formula1))
        self.wait(1)
        self.play(Write(formula2))
        self.wait(1)
        self.play(Write(formula3))
        self.wait(2)
        self.play(FadeOut(math_title), FadeOut(formulas))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: Attention preserves sequence length", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # === FAQ #6 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #6", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("Why research on context length if we can extend it?", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("Challenges of Extension", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("Quality degradation + O(S²) memory/compute costs", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept))

        # PART D: Diagram
        diagram_title = Text("Quality Degradation", font_size=24, color=RED).next_to(faq_header, DOWN, buff=0.3)
        self.play(FadeIn(diagram_title))
        
        # Quality bar
        bar_bg = Rectangle(width=8, height=0.5, color=GREY).shift(UP*1)
        trained_region = Rectangle(width=2, height=0.5, color=GREEN).align_to(bar_bg, LEFT).shift(UP*1)
        degradation_region = Rectangle(width=3, height=0.5, color=YELLOW).next_to(trained_region, RIGHT, buff=0).shift(UP*1)
        failure_region = Rectangle(width=3, height=0.5, color=RED).next_to(degradation_region, RIGHT, buff=0).shift(UP*1)
        
        label1 = Text("Trained", font_size=16).next_to(trained_region, UP, buff=0.1)
        label2 = Text("Degradation", font_size=16).next_to(degradation_region, UP, buff=0.1)
        label3 = Text("Failure", font_size=16).next_to(failure_region, UP, buff=0.1)
        
        self.play(Create(bar_bg))
        self.play(Create(trained_region), Write(label1))
        self.play(Create(degradation_region), Write(label2))
        self.play(Create(failure_region), Write(label3))
        self.wait(2)
        self.play(FadeOut(diagram_title), FadeOut(VGroup(bar_bg, trained_region, degradation_region, failure_region, label1, label2, label3)))

        # PART E: Math formula
        math_title = Text("Quadratic Costs", font_size=24, color=PURPLE).next_to(faq_header, DOWN, buff=0.3)
        formula1 = MathTex(r"\text{Memory} \propto S^2", font_size=28).next_to(math_title, DOWN)
        formula2 = MathTex(r"\text{Compute} \propto S^2", font_size=28).next_to(formula1, DOWN)
        formulas = VGroup(formula1, formula2)
        self.play(FadeIn(math_title), Write(formulas))
        self.wait(2)
        self.play(FadeOut(math_title), FadeOut(formulas))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: Extension causes severe quality & performance issues", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # === FAQ #7 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #7", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("What is RoPE and how do models achieve 1M context?", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("RoPE Mechanism", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("Rotary Positional Encoding rotates query/key vectors", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept))

        # PART E: Math formula
        math_title = Text("RoPE Rotation", font_size=24, color=PURPLE).next_to(faq_header, DOWN, buff=0.3)
        formula1 = MathTex(r"\text{angle} = m \times \theta_i", font_size=28).next_to(math_title, DOWN)
        formula2 = MathTex(r"\theta_i = \text{base}^{-2i/d}", font_size=28).next_to(formula1, DOWN)
        formulas = VGroup(formula1, formula2)
        self.play(FadeIn(math_title), Write(formulas))
        self.wait(2)
        self.play(FadeOut(math_title), FadeOut(formulas))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: RoPE enables relative position encoding", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # === FAQ #8 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #8", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("Explain RoPE math: R(m)@R(n).T = R(m-n)", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("Rotation Properties", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("Rotation matrices are orthogonal: R.T = R⁻¹", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept))

        # PART E: Math formula
        math_title = Text("Orthogonality Proof", font_size=24, color=PURPLE).next_to(faq_header, DOWN, buff=0.3)
        formula1 = MathTex(r"R(\theta)^T = R(-\theta)", font_size=28).next_to(math_title, DOWN)
        formula2 = MathTex(r"R(m) \cdot R(n)^T = R(m-n)", font_size=28).next_to(formula1, DOWN)
        formulas = VGroup(formula1, formula2)
        self.play(FadeIn(math_title), Write(formulas))
        self.wait(2)
        self.play(FadeOut(math_title), FadeOut(formulas))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: Attention depends only on relative positions", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # === FAQ #9 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #9", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("Explain torch.outer and freq concatenation", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("Frequency Computation", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("torch.outer creates position-frequency matrix", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept))

        # PART C: Code visualization
        code_title = Text("RoPE Cache Setup", font_size=24, color=BLUE).next_to(faq_header, DOWN, buff=0.3)
        code_line1 = Text("t = torch.arange(seq_len)", font_size=14, font="Monospace").next_to(code_title, DOWN, buff=0.2)
        code_line2 = Text("freqs = torch.outer(t, inv_freq)", font_size=14, font="Monospace").next_to(code_line1, DOWN, aligned_edge=LEFT)
        code_line3 = Text("emb = torch.cat((freqs, freqs), dim=-1)", font_size=14, font="Monospace").next_to(code_line2, DOWN, aligned_edge=LEFT)
        code_group = VGroup(code_line1, code_line2, code_line3)
        self.play(FadeIn(code_title))
        self.play(FadeIn(code_line1))
        self.wait(0.5)
        self.play(FadeIn(code_line2))
        self.wait(0.5)
        self.play(FadeIn(code_line3))
        self.wait(2)
        self.play(FadeOut(code_title), FadeOut(code_group))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: Outer product builds position-frequency matrix", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # === FAQ #10 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #10", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("Explain with simple example how RoPE works", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("Simple RoPE Example", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("4D vector with 2 rotation pairs at different frequencies", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept))

        # PART D: Diagram
        diagram_title = Text("Vector Pairing", font_size=24, color=RED).next_to(faq_header, DOWN, buff=0.3)
        self.play(FadeIn(diagram_title))
        
        vec_text = Text("q = [q₀, q₁, q₂, q₃]", font_size=24).shift(UP*1)
        pair1 = Text("Pair 1: (q₀, q₂) - Fast rotation (θ=1.0)", font_size=20).next_to(vec_text, DOWN, buff=0.5)
        pair2 = Text("Pair 2: (q₁, q₃) - Slow rotation (θ=0.01)", font_size=20).next_to(pair1, DOWN, buff=0.2)
        
        self.play(FadeIn(vec_text))
        self.play(FadeIn(pair1))
        self.play(FadeIn(pair2))
        self.wait(2)
        self.play(FadeOut(diagram_title), FadeOut(VGroup(vec_text, pair1, pair2)))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: Each pair rotates independently at its frequency", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # === FAQ #11 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #11", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("Won't model get confused with multiple rotations?", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("Independent Rotations", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("Frequencies operate on separate dimension pairs", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept))

        # PART D: Diagram
        diagram_title = Text("Separate Operations", font_size=24, color=RED).next_to(faq_header, DOWN, buff=0.3)
        self.play(FadeIn(diagram_title))
        
        # Dimension boxes
        dim_box = Rectangle(width=4, height=2, color=BLUE).shift(LEFT*2)
        dim_label = Text("4D Vector", font_size=18).next_to(dim_box, UP, buff=0.1)
        
        pair1_box = Rectangle(width=1.5, height=0.8, color=GREEN).shift(LEFT*3)
        pair1_label = Text("Pair 1", font_size=16).move_to(pair1_box)
        
        pair2_box = Rectangle(width=1.5, height=0.8, color=ORANGE).shift(LEFT*1)
        pair2_label = Text("Pair 2", font_size=16).move_to(pair2_box)
        
        freq1_box = Rectangle(width=1.5, height=0.8, color=YELLOW).shift(RIGHT*1)
        freq1_label = Text("θ=1.0", font_size=16).move_to(freq1_box)
        
        freq2_box = Rectangle(width=1.5, height=0.8, color=PURPLE).shift(RIGHT*3)
        freq2_label = Text("θ=0.01", font_size=16).move_to(freq2_box)
        
        # Arrows
        arrow1 = Arrow(pair1_box.get_right(), freq1_box.get_left())
        arrow2 = Arrow(pair2_box.get_right(), freq2_box.get_left())
        
        self.play(Create(dim_box), Write(dim_label))
        self.play(Create(pair1_box), Write(pair1_label))
        self.play(Create(pair2_box), Write(pair2_label))
        self.play(Create(arrow1), Create(freq1_box), Write(freq1_label))
        self.play(Create(arrow2), Create(freq2_box), Write(freq2_label))
        self.wait(2)
        self.play(FadeOut(diagram_title), FadeOut(VGroup(dim_box, dim_label, pair1_box, pair1_label, pair2_box, pair2_label, freq1_box, freq1_label, freq2_box, freq2_label, arrow1, arrow2)))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: Each frequency operates on its own dimension pair", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # === FAQ #12 ===
        self.clear()
        self.wait(0.3)

        # PART A: Question
        faq_header = Text("FAQ #12", font_size=36, color=YELLOW).to_edge(UP)
        question = Text("What do q₀, q₁, q₂, q₃ indicate in detail?", font_size=22).next_to(faq_header, DOWN, buff=0.3)
        self.play(Write(faq_header), FadeIn(question))
        self.wait(2)
        self.play(FadeOut(question))

        # PART B: Core Concept
        concept_title = Text("Dimension Components", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
        concept = Text("Each qᵢ represents one component of the query vector", font_size=20).next_to(concept_title, DOWN, buff=0.2)
        self.play(FadeIn(concept_title), FadeIn(concept))
        self.wait(2)
        self.play(FadeOut(concept_title), FadeOut(concept))

        # PART D: Diagram
        diagram_title = Text("Vector Components", font_size=24, color=RED).next_to(faq_header, DOWN, buff=0.3)
        self.play(FadeIn(diagram_title))
        
        # Vector representation
        vector_bracket = Text("[", font_size=48).shift(LEFT*2)
        q0 = Text("q₀", font_size=24, color=RED).next_to(vector_bracket, RIGHT, buff=0.1)
        comma1 = Text(",", font_size=24).next_to(q0, RIGHT, buff=0.1)
        q1 = Text("q₁", font_size=24, color=GREEN).next_to(comma1, RIGHT, buff=0.1)
        comma2 = Text(",", font_size=24).next_to(q1, RIGHT, buff=0.1)
        q2 = Text("q₂", font_size=24, color=BLUE).next_to(comma2, RIGHT, buff=0.1)
        comma3 = Text(",", font_size=24).next_to(q2, RIGHT, buff=0.1)
        q3 = Text("q₃", font_size=24, color=ORANGE).next_to(comma3, RIGHT, buff=0.1)
        close_bracket = Text("]", font_size=48).next_to(q3, RIGHT, buff=0.1)
        
        vector_elements = VGroup(vector_bracket, q0, comma1, q1, comma2, q2, comma3, q3, close_bracket)
        
        # Labels
        label0 = Text("Dim 0", font_size=16, color=RED).next_to(q0, DOWN, buff=0.3)
        label1 = Text("Dim 1", font_size=16, color=GREEN).next_to(q1, DOWN, buff=0.3)
        label2 = Text("Dim 2", font_size=16, color=BLUE).next_to(q2, DOWN, buff=0.3)
        label3 = Text("Dim 3", font_size=16, color=ORANGE).next_to(q3, DOWN, buff=0.3)
        
        self.play(FadeIn(vector_elements))
        self.play(FadeIn(label0), FadeIn(label1), FadeIn(label2), FadeIn(label3))
        self.wait(2)
        self.play(FadeOut(diagram_title), FadeOut(VGroup(vector_elements, label0, label1, label2, label3)))

        # PART F: Key takeaway
        takeaway = Text("Key Takeaway: Each qᵢ is a distinct dimension in the attention space", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
        self.play(FadeIn(takeaway))
        self.wait(2)
        self.clear()

        # Thank You Screen
        thank_you = Text("Thank You!", font_size=48, color=YELLOW).to_edge(UP)
        self.play(Write(thank_you))
        self.wait(3)