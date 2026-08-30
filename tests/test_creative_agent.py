import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from creative_agent import load_model_config, run_creative_pipeline
from creative_memory import DB_PATH, init_creative_memory_db, load_creative_preferences, save_creative_preference
from creative_models import CreativeRequest


def build_fake_ollama():
    revision_calls = {"count": 0}

    def fake_ollama_chat(*, model, messages):
        system_prompt = messages[0]["content"].lower()

        if "creative director" in system_prompt:
            idea_text = system_prompt
            if "whatsapp status" in idea_text:
                content = """
                {
                  "content_intent": "human contradiction comedy",
                  "emotional_tone": "wry",
                  "narrative_shape": "claim_to_reveal",
                  "use_psychology": true,
                  "use_philosophy": false,
                  "use_humor": true,
                  "use_ambiguity": true,
                  "humor_level": "strong",
                  "philosophy_level": "light",
                  "psychology_level": "high",
                  "ambiguity_level": "balanced",
                  "story_focus": "behavior reveals insecurity",
                  "rationale": "The request depends on contradiction, human truth, and observational humor."
                }
                """
            elif "bhagavad gita 3.38" in idea_text:
                content = """
                {
                  "content_intent": "philosophical reflection",
                  "emotional_tone": "quiet",
                  "narrative_shape": "metaphor_to_modern_life",
                  "use_psychology": true,
                  "use_philosophy": true,
                  "use_humor": false,
                  "use_ambiguity": true,
                  "humor_level": "off",
                  "philosophy_level": "deep",
                  "psychology_level": "medium",
                  "ambiguity_level": "balanced",
                  "story_focus": "preserve the metaphor and avoid preaching",
                  "rationale": "The request is reflective, source-based, and not comedic."
                }
                """
            else:
                content = """
                {
                  "content_intent": "educational explainer",
                  "emotional_tone": "clear",
                  "narrative_shape": "problem_to_solution",
                  "use_psychology": false,
                  "use_philosophy": false,
                  "use_humor": false,
                  "use_ambiguity": false,
                  "humor_level": "off",
                  "philosophy_level": "light",
                  "psychology_level": "light",
                  "ambiguity_level": "clear",
                  "story_focus": "clarity over flourish",
                  "rationale": "The request is straightforward and does not need every specialist."
                }
                """
        elif "ordinary human behavior" in system_prompt:
            content = """
            {
              "visible_behavior": "He says he is above public opinion while checking reactions constantly.",
              "hidden_motive": "He wants reassurance and status confirmation.",
              "emotional_trigger": "Social comparison.",
              "contradiction": "He performs indifference while chasing approval.",
              "audience_rel_path": "Most viewers have seen this behavior in themselves or others."
            }
            """
        elif "philosophical tension" in system_prompt:
            content = """
            {
              "central_question": "What covers insight even when truth is present?",
              "deeper_meaning": "Desire can obscure perception without destroying the underlying truth.",
              "tension": "Truth remains, but access to it becomes clouded.",
              "possible_closing_thought": "Maybe the task is not creating truth but clearing what covers it.",
              "avoid_preaching": "Prefer observation and question over instruction.",
              "source_meaning": "The verse describes desire veiling discernment through layered metaphors.",
              "modern_reflection": "Modern life adds distraction, vanity, and restless craving to that veil."
            }
            """
        elif "preserve ambiguity" in system_prompt:
            content = """
            {
              "competing_interpretations": [
                "He is insecure.",
                "He is simply curious about who noticed him."
              ],
              "unresolved_question": "Does he know how much he needs the validation?",
              "contradiction": "His self-story and his behavior do not match.",
              "what_not_to_explain": "Do not explain his motive in a final moral sentence.",
              "ambiguity_strength": "balanced"
            }
            """
        elif "generate humor from recognizable human truth" in system_prompt:
            content = """
            {
              "humor_style": "observational",
              "setup": "He says other opinions mean nothing.",
              "punchline_candidates": [
                "Five minutes later he is holding a refresh button like a prayer bead.",
                "Apparently indifference now comes with read receipts.",
                "He does not care what people think, just who exactly thought it."
              ],
              "callback_candidate": "Another quick glance at the viewer list.",
              "awkward_truth": "People often reject approval with words and seek it with rituals.",
              "avoid_cruelty": "Keep the joke on the contradiction, not on protected traits."
            }
            """
        elif "you are the story writer" in system_prompt:
            if "bhagavad gita 3.38" in system_prompt:
                content = """
                {
                  "premise": "Bhagavad Gita 3.38 names three coverings that hide clarity before turning toward modern desire.",
                  "conflict": "Clarity remains present, but desire keeps obscuring access to it.",
                  "progression": "Smoke covers fire, dust obscures a mirror, unborn life remains enclosed within the womb, and only then does the scene move toward a modern mind clouded by craving.",
                  "emotional_turn": "A modern person recognizes that mental restlessness has covered simple clarity.",
                  "ending": "The disturbance settles and perception quietly clears.",
                  "scene_beats": [
                    "Smoke partially covers a fire.",
                    "Dust obscures a mirror.",
                    "Unborn life remains enclosed within the womb.",
                    "A modern person sits in thought as inner clarity begins to feel clouded.",
                    "Desire and comparison subtly cloud the mind.",
                    "The disturbance settles and clarity quietly returns."
                  ]
                }
                """
            else:
                content = """
                {
                  "premise": "A man tries to look above public opinion while orbiting his own status viewers.",
                  "conflict": "His self-image and his behavior keep colliding.",
                  "progression": "Each refresh turns his claim of indifference into a visual confession.",
                  "emotional_turn": "He notices the contradiction in real time.",
                  "ending": "The final silence lands better than an explanation.",
                  "scene_beats": [
                    "He says he does not care what people think.",
                    "He checks the WhatsApp status viewer list again.",
                    "He catches himself and says nothing."
                  ]
                }
                """
        elif "final editor and critic" in system_prompt:
            content = """
            {
              "relatability_score": 9,
              "humor_score": 8,
              "psychological_truth_score": 9,
              "philosophical_depth_score": 3,
              "ambiguity_score": 7,
              "preachiness_score": 2,
              "originality_score": 7,
              "clarity_score": 8,
              "forced_humor_score": 1,
              "unnecessary_explanation_score": 2,
              "notes": "The contradiction is strong; trim any line that explains the joke too directly.",
              "edit_instructions": [
                "Let the phone-checking action carry the reveal.",
                "End on the quiet recognition instead of a moral."
              ]
            }
            """
        elif "revise the story exactly once" in system_prompt:
            revision_calls["count"] += 1
            if "bhagavad gita 3.38" in system_prompt:
                content = """
                {
                  "premise": "Bhagavad Gita 3.38 shows desire as a covering over clarity through three concrete metaphors before turning toward modern restlessness.",
                  "conflict": "Human clarity is present, yet desire and craving keep veiling it.",
                  "progression": "Smoke covers fire, dust obscures a mirror, unborn life stays enclosed within the womb, then a modern mind grows clouded by comparison and wanting.",
                  "emotional_turn": "The person notices the mental noise and begins to let it settle.",
                  "ending": "Clarity quietly returns without spectacle.",
                  "scene_beats": [
                    "Smoke partially covers a fire.",
                    "Dust obscures a mirror.",
                    "Unborn life remains enclosed within the womb.",
                    "A modern person sits in thought as inner clarity begins to feel clouded.",
                    "Desire and comparison subtly cloud the mind.",
                    "The disturbance settles and clarity quietly returns."
                  ]
                }
                """
            else:
                content = """
                {
                  "premise": "A man says he does not care what people think, then keeps checking who viewed his WhatsApp status.",
                  "conflict": "His words protect his pride while his actions betray his need for approval.",
                  "progression": "The refreshes get quicker and harder to excuse.",
                  "emotional_turn": "He sees himself doing it and cannot hide behind the speech anymore.",
                  "ending": "He locks the phone, but not before one last glance.",
                  "scene_beats": [
                    "Public claim of indifference.",
                    "Private ritual of checking.",
                    "Quiet self-recognition."
                  ]
                }
                """
        elif "extract only the narrative requirements" in system_prompt:
            if "bhagavad gita 3.38" in system_prompt:
                content = """
                {
                  "narrative_constraints": [
                    {
                      "id": "constraint_1",
                      "constraint_type": "source_metaphor",
                      "description": "smoke covering fire",
                      "importance": "required",
                      "source_order": 1
                    },
                    {
                      "id": "constraint_2",
                      "constraint_type": "source_metaphor",
                      "description": "dust covering a mirror",
                      "importance": "required",
                      "source_order": 2
                    },
                    {
                      "id": "constraint_3",
                      "constraint_type": "source_metaphor",
                      "description": "unborn life enclosed within the womb",
                      "importance": "required",
                      "source_order": 3
                    }
                  ]
                }
                """
            elif "online purchase" in system_prompt or "package" in system_prompt:
                content = """
                {
                  "narrative_constraints": [
                    {
                      "id": "constraint_1",
                      "constraint_type": "plot_event",
                      "description": "husband hides an online purchase",
                      "importance": "required",
                      "source_order": 1
                    },
                    {
                      "id": "constraint_2",
                      "constraint_type": "plot_event",
                      "description": "wife finds the package",
                      "importance": "required",
                      "source_order": 2
                    },
                    {
                      "id": "constraint_3",
                      "constraint_type": "plot_event",
                      "description": "husband pretends it was already in the house",
                      "importance": "required",
                      "source_order": 3
                    }
                  ]
                }
                """
            elif "evaporation" in system_prompt and "condensation" in system_prompt:
                content = """
                {
                  "narrative_constraints": [
                    {
                      "id": "constraint_1",
                      "constraint_type": "educational_step",
                      "description": "evaporation",
                      "importance": "required",
                      "source_order": 1
                    },
                    {
                      "id": "constraint_2",
                      "constraint_type": "educational_step",
                      "description": "condensation",
                      "importance": "required",
                      "source_order": 2
                    },
                    {
                      "id": "constraint_3",
                      "constraint_type": "educational_step",
                      "description": "rainfall",
                      "importance": "required",
                      "source_order": 3
                    }
                  ]
                }
                """
            else:
                content = """
                {
                  "narrative_constraints": [
                    {
                      "id": "constraint_1",
                      "constraint_type": "contradiction",
                      "description": "He performs indifference while chasing approval.",
                      "importance": "required",
                      "source_order": null
                    }
                  ]
                }
                """
        else:
            raise AssertionError(f"Unexpected prompt: {system_prompt}")

        return {"message": {"content": content}}

    fake_ollama_chat.revision_calls = revision_calls
    return fake_ollama_chat


class CreativeAgentTests(unittest.TestCase):
    def test_director_selects_humor_for_human_contradiction_request(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea="A man says he doesn't care what people think, but checks who viewed his WhatsApp status every five minutes.",
            content_type="humor",
            tone="wry",
            target_audience="general",
            language="English",
            duration_seconds=15,
            visual_style="Quiet Cinematic Animation",
            humor_level="strong",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)
        self.assertIn("humor", result.selected_specialists)
        self.assertIn("psychology", result.selected_specialists)
        self.assertIn("ambiguity", result.selected_specialists)

    def test_director_selects_philosophy_for_source_based_reflection(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea="Create a quiet animated reflection on Bhagavad Gita 3.38 using smoke covering fire, dust covering a mirror, and the womb enclosing unborn life.",
            content_type="philosophy",
            tone="quiet",
            target_audience="general",
            language="English",
            duration_seconds=30,
            visual_style="Quiet Cinematic Animation",
            depth_level="deep",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)
        self.assertIn("philosophy", result.selected_specialists)
        self.assertIn("psychology", result.selected_specialists)
        self.assertIn("ambiguity", result.selected_specialists)
        self.assertNotIn("humor", result.selected_specialists)
        self.assertEqual(result.philosophy.source_meaning.startswith("The verse describes desire"), True)

    def test_verse_based_reflection_activates_required_specialists_and_preserves_metaphors(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea=(
                "Create a quiet animated reflection on Bhagavad Gita 3.38 using smoke covering fire, "
                "dust covering a mirror, and the womb enclosing unborn life. Use those metaphors to "
                "explore how desire can obscure human clarity."
            ),
            content_type="spiritual_reflection",
            tone="quiet reflective",
            target_audience="general",
            language="English",
            duration_seconds=30,
            visual_style="Quiet Cinematic Animation",
            depth_level="deep",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)

        self.assertEqual(
            result.selected_specialists,
            ["psychology", "philosophy", "ambiguity", "story", "critic"],
        )
        self.assertEqual(
            result.final_story.scene_beats[:3],
            [
                "Smoke partially covers a fire.",
                "Dust obscures a mirror.",
                "Unborn life remains enclosed within the womb.",
            ],
        )
        self.assertIn("modern person", result.final_story.scene_beats[3].lower())

    def test_director_does_not_select_all_specialists_for_education(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea="Explain compound interest for beginners in 15 seconds.",
            content_type="education",
            tone="clear",
            target_audience="general",
            language="English",
            duration_seconds=15,
            visual_style="Minimal Illustration",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)
        self.assertEqual(result.selected_specialists, ["story", "critic"])

    def test_psychology_output_returns_structured_contradiction_without_diagnosis(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea="A man says he doesn't care what people think, but checks who viewed his WhatsApp status every five minutes.",
            content_type="humor",
            tone="wry",
            target_audience="general",
            language="English",
            duration_seconds=15,
            visual_style="Quiet Cinematic Animation",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)
        self.assertTrue(result.psychology.contradiction)
        self.assertNotIn("disorder", result.psychology.hidden_motive.lower())

    def test_humor_returns_multiple_candidates(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea="A man says he doesn't care what people think, but checks who viewed his WhatsApp status every five minutes.",
            content_type="humor",
            tone="wry",
            target_audience="general",
            language="English",
            duration_seconds=15,
            visual_style="Quiet Cinematic Animation",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)
        self.assertEqual(result.humor.humor_style, "observational")
        self.assertGreaterEqual(len(result.humor.punchline_candidates), 3)

    def test_ambiguity_returns_competing_interpretations(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea="A man says he doesn't care what people think, but checks who viewed his WhatsApp status every five minutes.",
            content_type="human_behavior",
            tone="balanced",
            target_audience="general",
            language="English",
            duration_seconds=15,
            visual_style="Quiet Cinematic Animation",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)
        self.assertGreaterEqual(len(result.ambiguity.competing_interpretations), 2)

    def test_story_uses_specialist_insight_and_revision_runs_once(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea="A man says he doesn't care what people think, but checks who viewed his WhatsApp status every five minutes.",
            content_type="humor",
            tone="wry",
            target_audience="general",
            language="English",
            duration_seconds=15,
            visual_style="Quiet Cinematic Animation",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)
        self.assertIn("WhatsApp status", result.final_story.premise)
        self.assertEqual(fake_ollama.revision_calls["count"], 1)

    def test_gita_request_produces_only_three_source_constraints(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea=(
                "Create a 30-second quiet cinematic reflection on Bhagavad Gita 3.38.\n\n"
                "The verse uses three metaphors:\n"
                "- smoke covering fire\n"
                "- dust covering a mirror\n"
                "- unborn life enclosed by the womb\n\n"
                "Use those metaphors to explore how desire can obscure human clarity.\n\n"
                "Keep the scriptural meaning separate from the modern psychological interpretation.\n"
                "Do not force humor.\n"
                "Keep some ambiguity.\n"
                "Use quiet cinematic animation with muted earthy colors and restrained movement.\n\n"
                "Do not force the same character into symbolic scenes.\n"
                "Decide the best scene structure yourself."
            ),
            content_type="spiritual_reflection",
            tone="quiet reflective",
            target_audience="general",
            language="English",
            duration_seconds=30,
            visual_style="Quiet Cinematic Animation",
            depth_level="deep",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)
        self.assertEqual(
            [constraint.description for constraint in result.narrative_constraints],
            [
                "smoke covering fire",
                "dust covering a mirror",
                "unborn life enclosed within the womb",
            ],
        )
        joined = " | ".join(constraint.description for constraint in result.narrative_constraints).lower()
        self.assertNotIn("restrained movement", joined)
        self.assertNotIn("same character", joined)

    def test_plot_request_constraints_exclude_style_language(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea=(
                "Create a funny story using a husband hiding an online purchase, his wife finding the package, "
                "and his attempt to pretend it was already in the house. Use warm lighting and restrained animation."
            ),
            content_type="humor",
            tone="wry",
            target_audience="general",
            language="English",
            duration_seconds=15,
            visual_style="Quiet Cinematic Animation",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)
        self.assertEqual(
            [constraint.description for constraint in result.narrative_constraints],
            [
                "husband hides an online purchase",
                "wife finds the package",
                "husband pretends it was already in the house",
            ],
        )

    def test_educational_request_constraints_exclude_visual_style_language(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea="Create an educational video using evaporation, condensation, and rainfall. Use simple illustrations and slow motion.",
            content_type="education",
            tone="clear",
            target_audience="general",
            language="English",
            duration_seconds=15,
            visual_style="Minimal Illustration",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)
        self.assertEqual(
            [constraint.description for constraint in result.narrative_constraints],
            ["evaporation", "condensation", "rainfall"],
        )

    def test_critic_produces_structured_scores(self) -> None:
        fake_ollama = build_fake_ollama()
        request = CreativeRequest(
            idea="A man says he doesn't care what people think, but checks who viewed his WhatsApp status every five minutes.",
            content_type="humor",
            tone="wry",
            target_audience="general",
            language="English",
            duration_seconds=15,
            visual_style="Quiet Cinematic Animation",
        )
        result = run_creative_pipeline(request=request, ollama_chat=fake_ollama)
        self.assertEqual(result.critic.relatability_score, 9)
        self.assertEqual(len(result.critic.edit_instructions), 2)

    def test_model_config_defaults_and_env_overrides_work(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            defaults = load_model_config()
            self.assertEqual(defaults.director_model, "qwen3:8b")
            self.assertEqual(defaults.humor_model, "qwen3:8b")

        with patch.dict(
            os.environ,
            {
                "AUMSTATE_DIRECTOR_MODEL": "qwen3:8b",
                "AUMSTATE_HUMOR_MODEL": "llama3.1:8b",
                "AUMSTATE_CRITIC_MODEL": "mistral:7b",
            },
            clear=True,
        ):
            config = load_model_config()
            self.assertEqual(config.director_model, "qwen3:8b")
            self.assertEqual(config.humor_model, "llama3.1:8b")
            self.assertEqual(config.critic_model, "mistral:7b")


class CreativeMemoryTests(unittest.TestCase):
    def test_creative_preferences_persist_without_affecting_chat_memory_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_db = str(Path(temp_dir) / "creative_memory.db")
            with patch("creative_memory.DB_PATH", temp_db):
                init_creative_memory_db(db_path=temp_db)
                save_creative_preference("preferred_humor_style", "observational", db_path=temp_db)
                preferences = load_creative_preferences(db_path=temp_db)
                self.assertEqual(preferences["preferred_humor_style"], "observational")
                self.assertEqual(DB_PATH, "aumstate_memory.db")
