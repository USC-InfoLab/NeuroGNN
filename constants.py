# Channels of interest
INCLUDED_CHANNELS = [
    'EEG FP1',
    'EEG FP2',
    'EEG F3',
    'EEG F4',
    'EEG C3',
    'EEG C4',
    'EEG P3',
    'EEG P4',
    'EEG O1',
    'EEG O2',
    'EEG F7',
    'EEG F8',
    'EEG T3',
    'EEG T4',
    'EEG T5',
    'EEG T6',
    'EEG FZ',
    'EEG CZ',
    'EEG PZ']

# Resampling frequency
FREQUENCY = 200

# All seizure labels available in TUH
ALL_LABEL_DICT = {'fnsz': 0, 'gnsz': 1, 'spsz': 2, 'cpsz': 3,
                  'absz': 4, 'tnsz': 5, 'tcsz': 6, 'mysz': 7}



   
CORTEX_REGIONS = [
    'Pre-Frontal Lobe',
    'Frontal Lobe',
    'Parietal Lobe',
    'Temporal Lobe',
    'Occipital Lobe',
    'Central Gyrus'
]



ELECTRODES_BROADMANN_MAPPING = {
    'EEG FP1': 'BA10 Left',
    'EEG FP2': 'BA10 Right',
    'EEG F3': 'BA9 Left',
    'EEG F4': 'BA9 Right',
    'EEG C3': 'BA1 Left',
    'EEG C4': 'BA1 Right',
    'EEG P3': 'BA39 Left',
    'EEG P4': 'BA39 Right',
    'EEG O1': 'BA18 Left',
    'EEG O2': 'BA18 Right',
    'EEG F7': 'BA45 Left',
    'EEG F8': 'BA45 Right',
    'EEG T3': 'BA42/22 Left',
    'EEG T4': 'BA42/22 Right',
    'EEG T5': 'BA19 Left',
    'EEG T6': 'BA19 Right',
    'EEG FZ': 'BA6 Left',
    'EEG CZ': 'BA4 Right',
    'EEG PZ': 'BA7 Left'
}




# Source: Wikipedia - For central gyrus, both precentral and postcentral are used. Source for postcentral gyrus:  https://www.ncbi.nlm.nih.gov/books/NBK549825/
CORTEX_REGIONS_DESCRIPTIONS = {
    'Pre-Frontal Lobe': """
                        The basic activity of pre-frontal lobe brain region is considered to be orchestration of thoughts and actions in accordance with internal goals. Many authors have indicated an integral link between a person's will to live, personality, and the functions of the prefrontal cortex.
                        This brain region has been implicated in executive functions, such as planning, decision making, working memory, personality expression, moderating social behavior and controlling certain aspects of speech and language. Executive function relates to abilities to differentiate among conflicting thoughts, determine good and bad, better and best, same and different, future consequences of current activities, working toward a defined goal, prediction of outcomes, expectation based on actions, and social "control" (the ability to suppress urges that, if not suppressed, could lead to socially unacceptable outcomes).
                        The frontal cortex supports concrete rule learning. More anterior regions along the rostro-caudal axis of frontal cortex support rule learning at higher levels of abstraction.
                        """,
    'Frontal Lobe': """
                    The frontal lobe is covered by the frontal cortex. The frontal cortex includes the premotor cortex and the primary motor cortex, parts of the motor cortex. The front part of the frontal cortex is covered by the prefrontal cortex. The nonprimary motor cortex is a functionally defined portion of the frontal lobe.
                    There are four principal gyri in the frontal lobe. The precentral gyrus is directly anterior to the central sulcus, running parallel to it and contains the primary motor cortex, which controls voluntary movements of specific body parts. Three horizontally arranged subsections of the frontal gyrus are the superior frontal gyrus, the middle frontal gyrus, and the inferior frontal gyrus. The inferior frontal gyrus is divided into three parts the orbital part, the triangular part and the opercular part.
                    The frontal lobe contains most of the dopamine neurons in the cerebral cortex. The dopaminergic pathways are associated with reward, attention, short-term memory tasks, planning, and motivation. Dopamine tends to limit and select sensory information coming from the thalamus to the forebrain.
                    """,
    'Parietal Lobe': """
                    The parietal lobe integrates sensory information among various modalities, including spatial sense and navigation (proprioception), the main sensory receptive area for the sense of touch in the somatosensory cortex which is just posterior to the central sulcus in the postcentral gyrus, and the dorsal stream of the visual system. The major sensory inputs from the skin (touch, temperature, and pain receptors), relay through the thalamus to the parietal lobe.
                    Several areas of the parietal lobe are important in language processing. The somatosensory cortex can be illustrated as a distorted figure, the cortical homunculus (Latin: "little man") in which the body parts are rendered according to how much of the somatosensory cortex is devoted to them. The superior parietal lobule and inferior parietal lobule are the primary areas of body or spatial awareness. A lesion commonly in the right superior or inferior parietal lobule leads to hemineglect.
                    """,
    'Temporal Lobe': """
                    The temporal lobe is involved in processing sensory input into derived meanings for the appropriate retention of visual memory, language comprehension, and emotion association. Temporal refers to the head's temples.

                    Visual memories
                    The temporal lobe communicates with the hippocampus and plays a key role in the formation of explicit long-term memory modulated by the amygdala.

                    Processing sensory input Auditory
                    Adjacent areas in the superior, posterior, and lateral parts of the temporal lobes are involved in high-level auditory processing. The temporal lobe is involved in primary auditory perception, such as hearing, and holds the primary auditory cortex. The primary auditory cortex receives sensory information from the ears and secondary areas process the information into meaningful units such as speech and words. The superior temporal gyrus includes an area (within the lateral fissure) where auditory signals from the cochlea first reach the cerebral cortex and are processed by the primary auditory cortex in the left temporal lobe.

                    Visual
                    The areas associated with vision in the temporal lobe interpret the meaning of visual stimuli and establish object recognition. The ventral part of the temporal cortices appear to be involved in high-level visual processing of complex stimuli such as faces (fusiform gyrus) and scenes (parahippocampal gyrus). Anterior parts of this ventral stream for visual processing are involved in object perception and recognition.

                    Language recognition
                    The temporal lobe holds the primary auditory cortex, which is important for the processing of semantics in both language and vision in humans. Wernicke's area, which spans the region between temporal and parietal lobes, plays a key role (in tandem with Broca's area in the frontal lobe) in language comprehension, whether spoken language or signed language. FMRI imaging shows these portions of the brain are activated by signed or spoken languages. These areas of the brain are active in children's language acquisition whether accessed via hearing a spoken language, watching a signed language, or via hand-over-hand tactile versions of a signed language.
                    The functions of the left temporal lobe are not limited to low-level perception but extend to comprehension, naming, and verbal memory.

                    New memories
                    The medial temporal lobes (near the sagittal plane) are thought to be involved in encoding declarative long term memory. The medial temporal lobes include the hippocampi, which are essential for memory storage, therefore damage to this area can result in impairment in new memory formation leading to permanent or temporary anterograde amnesia.                
                    """,
    'Occipital Lobe': """
                    The occipital lobe is the visual processing center of the mammalian brain containing most of the anatomical region of the visual cortex. The primary visual cortex is Brodmann area 17, commonly called V1 (visual one). Human V1 is located on the medial side of the occipital lobe within the calcarine sulcus; the full extent of V1 often continues onto the occipital pole. V1 is often also called striate cortex because it can be identified by a large stripe of myelin, the Stria of Gennari. Visually driven regions outside V1 are called extrastriate cortex. There are many extrastriate regions, and these are specialized for different visual tasks, such as visuospatial processing, color differentiation, and motion perception. Bilateral lesions of the occipital lobe can lead to cortical blindness.    
                    """,
    'Central Gyrus': """
                    Contains both the precentral gyrus and postcentral gyrus.
                    precentral gyrus: 
                    The precentral gyrus is specialised for sending signals down to the spinal cord for movement. As they travel down through the cerebral white matter, the motor axons move closer together and form part of the posterior limb of the internal capsule. They continue down into the brainstem, where some of them, after crossing over to the contralateral side, distribute to the cranial nerve motor nuclei. (Note: a few motor fibers synapse with lower motor neurons on the same side of the brainstem). After crossing over to the contralateral side in the medulla oblongata (pyramidal decussation), the axons travel down the spinal cord as the lateral corticospinal tract. Fibers that do not cross over in the brainstem travel down the separate ventral corticospinal tract and most of them cross over to the contralateral side in the spinal cord, shortly before reaching the lower motor neurons.

                    postcentral gyrus:
                    The postcentral gyrus is found on the lateral surface of the anterior parietal lobe, caudal to the central sulcus, and corresponds to Brodmann areas 3b, 1, and 2. The primary somatosensory cortex perceives sensations on the contralateral side. The topographic organization of this region is known as the sensory homunculus, or “little man.” This organization of the somatosensory map is such that the medial aspect is responsible for lower extremity sensation, the dorsolateral aspect is responsible for the upper extremity, and the most lateral aspect is responsible for the face, lips, and tongue. However, regions of the homunculus that require high sensory acuity and resolution take up a larger area on the somatosensory map. For example, the hands, face, and lips necessitate fine somatosensory perception relative to other regions, such as the leg or torso. The postcentral gyrus also houses the secondary somatosensory cortex, which appears to play a role in the integration of somatosensory stimuli and memory formation.
                    """,
}


ELECTRODES_REGIONS = {
    'EEG FP1': 'Pre-Frontal Lobe',
    'EEG FP2': 'Pre-Frontal Lobe',
    'EEG F3': 'Frontal Lobe',
    'EEG F4': 'Frontal Lobe',
    'EEG C3': 'Central Gyrus',
    'EEG C4': 'Central Gyrus',
    'EEG P3': 'Parietal Lobe',
    'EEG P4': 'Parietal Lobe',
    'EEG O1': 'Occipital Lobe',
    'EEG O2': 'Occipital Lobe',
    'EEG F7': 'Frontal Lobe',
    'EEG F8': 'Frontal Lobe',
    'EEG T3': 'Temporal Lobe',
    'EEG T4': 'Temporal Lobe',
    'EEG T5': 'Temporal Lobe',
    'EEG T6': 'Temporal Lobe',
    'EEG FZ': 'Frontal Lobe',
    'EEG CZ': 'Central Gyrus',
    'EEG PZ': 'Parietal Lobe'
}


meta_node_indices = {}

for cortex_region in CORTEX_REGIONS:
    meta_node_indices[cortex_region] = []

for i, node in enumerate(INCLUDED_CHANNELS):
    cortex_region = ELECTRODES_REGIONS[node]
    meta_node_indices[cortex_region].append(i)

META_NODE_INDICES = list(meta_node_indices.values())



# Source: http://www.stresstherapysolutions.com/uploads/Brodmann-Detail-Genardi.pdf
BROADMANN_AREA_DESCRIPTIONS = {
    'BA10 Left': """
                Part of the prefrontal cortex - Middle frontal gyrus
                Memory:
                Working memory
                Spatial memory
                Memory encoding and recognition
                Memory retrieval
                Event- and time-based prospective memory
                Prospective memory
                Intentional forgetting

                Language:
                Syntactic processing
                Metaphor comprehension
                Word-stem completion
                Verb generation

                Auditory:
                Nonspeech processing (monaural stimulus)

                Other:
                Processing emotional stimuli
                Processing emotions and self-reflections in decision making
                Calculation / numerical processes
                Intention/sensory feedback conflict detection
                Pleasant and unpleasant emotions
                Response to painful thermal stimuli
                Joint attention
                """,
    'BA10 Right': """
                Part of the prefrontal cortex - Middle frontal gyrus
                Memory:
                Working memory
                Spatial memory
                Memory encoding and recognition
                Memory retrieval
                Event- and time-based prospective memory
                Prospective memory
                Intentional forgetting

                Auditory:
                Nonspeech processing (monaural stimulus)

                Other:
                Processing emotional stimuli
                Decision making (involving conflict and reward)
                Calculation / numerical processes
                Intention/sensory feedback conflict detection
                Smelling familiar odors
                Pleasant and unpleasant emotions
                Response to painful thermal stimuli
                Joint attention
                """,
    'BA9 Left': """
                Part of the prefrontal cortex - Middle frontal gyrus
                Memory:
                Working memory
                Spatial memory
                Short-term memory
                Memory encoding and recognition
                Memory retrieval
                Recency judgments

                Motor:
                Executive control of behavior

                Language:
                Syntactic processing
                Metaphor comprehension
                Verbal fluency
                Semantic categorization
                Word-stem completion
                Generating sentences

                Other:
                Error processing/detection
                Attention to human voices
                Processing emotional stimuli
                Processing emotions and self-reflections in decision making
                Inferential reasoning
                Calculation / numerical processes
                Attribution of intention to others
                Intention/sensory feedback conflict detection
                Pleasant and unpleasant emotions
                """,
    'BA9 Right': """
                Part of the prefrontal cortex - Middle frontal gyrus
                Memory:
                Working memory
                Spatial memory
                Short-term memory
                Memory encoding and recognition
                Memory retrieval
                Recency judgments

                Motor:
                Executive control of behavior


                Other:
                Error processing/detection
                Attention to human voices
                Processing emotional stimuli
                Inferential reasoning
                Planning
                Calculation / numerical processes
                Attribution of intention to others
                Intention/sensory feedback conflict detection
                Smelling familiar odors
                Pleasant and unpleasant emotions
                """,
    'BA1 Left': """
                Primary somatosensory cortex - Postcentral gyrus
                Somatosensory:
                Localization of touch
                Localization of temperature
                Localization of vibration
                Localization of pain
                Finger proprioception
                Voluntary hand movement
                Volitional swallowing
                Skillful coordinated orofacial movement (i.e. whistling)

                Other
                Somatosensory mirror neuron system
                Touch anticipation (i.e. tickling)
                Mirror neurons for speech perception
                Motor learning
                """,
    'BA1 Right': """
                Primary somatosensory cortex - Postcentral gyrus
                Somatosensory:
                Localization of touch
                Localization of temperature
                Localization of vibration
                Localization of pain
                Finger proprioception
                Voluntary hand movement
                Volitional swallowing
                Skillful coordinated orofacial movement (i.e. whistling)

                Other
                Somatosensory mirror neuron system
                Touch anticipation (i.e. tickling)
                Mirror neurons for speech perception
                Motor learning
                """,
    'BA39 Left': """
                Part of inferior parietal lobule - Caudal bank of intraparietal sulcus -
                Angular gyrus
                Part of Wernicke's area
                Language:
                Sentence generation
                Reading

                Calculation:
                Calculation
                Arithmetic learning
                Abstract coding of numerical magnitude

                Visual:
                Spatial focusing of attention

                Other:
                Performingverbal creative tasks
                Theory of mind
                Executive control of behavior
                Processing a sequence of actions
                """,
    'BA39 Right': """
                Part of inferior parietal lobule - Caudal bank of intraparietal sulcus -
                Angular gyrus
                Part of Wernicke's area
                Language:
                Sentence generation
                Reading

                Visual:
                Spatial focusing of attention
                Visuospatial processing

                Other:
                Theory of mind
                Executive control of behavior
                Sight reading (music)
                """,
    'BA18 Left': """ 
                Secondary visual cortex - Middle occipital gyrus
                Visual:
                Detection of light intensity
                Detection of patterns
                Tracking visual motion patterns (optokinetic stimulation)
                Discrimination of finger gestures
                Sustained attention to color and shape
                Feature-based attention
                Orientation-selective attention

                Memory:
                Visual priming
                Word and face encoding

                Language:
                Response to visual word form
                Confrontation naming

                Other:
                Face-name association
                Horizontal saccadic eye movements
                Visual mental imagery
                """,
    'BA18 Right': """
                Secondary visual cortex - Middle occipital gyrus
                Visual:
                Detection of light intensity
                Detection of patterns
                Tracking visual motion patterns (optokinetic stimulation)
                Discrimination of finger gestures
                Sustained attention to color and shape
                Visuo-spatial information processing
                Feature-based attention
                Orientation-selective attention

                Memory:
                Visual priming
                Word and face encoding

                Language:
                Confrontation naming

                Other:
                Horizontal saccadic eye movements
                Response to emotion/attention in visual processing
                """,
    'BA45 Left': """
                Broca's Area
                Inferior frontal gyrus - Pars triangularis
                Language:
                Semantic and phonological processing
                Internally specified word generation
                Verbal fluency
                Lexical search
                Phonological processing
                Grammatical processing
                Semantic memory retrieval
                Selective attention to speech
                Sign language
                Lexical inflection
                Reasoning processes
                Processing of metaphors

                Memory:
                Working memory
                Non-verbal working memory (bilaterally)
                Episodic long-term memory
                Declarative memory encoding
                Recall of digit series

                Motor:
                Mirror neurons for expressive movements
                Mirror neurons for grasping movements
                Response inhibition

                Other:
                Mental rotation (mostly in females)
                Word and face encoding
                Aesthetic appreciation
                Music enjoyment
                Generation of melodic phrases
                Modulating emotional response
                Smelling familiar odors
                """,
    'BA45 Right': """
                Broca's Area
                Inferior frontal gyrus - Pars triangularis
                Language:
                Semantic and phonological processing
                Internally specified word generation
                Verbal fluency
                Lexical search
                Phonological processing
                Grammatical processing
                Semantic memory retrieval
                Sign language
                Affective prosody comprehension
                Reasoning processes
                Processing of metaphors

                Memory:
                Working memory
                Non-verbal working memory (bilaterally)
                Episodic long-term memory
                Declarative memory encoding
                Recall of digit series

                Motor:
                Mirror neurons for expressive movements
                Mirror neurons for grasping movements
                Response inhibition

                Other:
                Mental rotation (mostly in females)
                Word and face encoding
                Aesthetic appreciation
                Music enjoyment
                Modulating emotional response
                """,
    'BA42/22 Left': """ 
                    Primary auditory cortex - Heschl's gyrus
                    Superior Temporal Gyrus - Part of Wernicke's area

                    Auditory:
                    Basic processing of auditory stimuli (speech and non-speech)
                    Processing discontinued acoustic patterns
                    Frequency deviant detection
                    Perception of harmonic tones
                    Processing sound intensity
                    Sensitivity to pitch
                    Rapid sound detection (Bilateral)
                    Sound (vowel) segregation
                    Auditory priming
                    Processing complex sounds
                    Lexico-semantic access to melodic representations (Anterior)

                    Receptive language:
                    Auditory language processing
                    Semantic processing
                    Sentence generation
                    Frequency deviant detection
                    Internally-specified word generation

                    Language-related:
                    Selective attention to speech
                    Learning a tone-based second language
                    Repeating words

                    Memory:
                    Repetition priming effect
                    Auditory working memory

                    Other:
                    Visual speech perception (mirror neurons?)
                    Attribution of intentions to others
                    Deductive reasoning
                    """,
    'BA42/22 Right': """
                    Primary auditory cortex - Heschl's gyrus
                    Superior Temporal Gyrus - Part of Wernicke's area
                    Auditory:
                    Basic processing of auditory stimuli (speech and non-speech)
                    Processing discontinued acoustic patterns
                    Frequency deviant detection
                    Perception of harmonic tones
                    Processing sound intensity
                    Sensitivity to pitch
                    Rapid sound detection (Bilateral)
                    Sound (vowel) segregation
                    Auditory priming
                    Nonverbal sounds processing
                    Processing complex sounds
                    Lexico-semantic access to melodic representations (Anterior)

                    Receptive language:
                    Sentence generation
                    Frequency deviant detection

                    Language-related:
                    Selective attention to speech
                    Affective prosody comprehension
                    Repeating words

                    Visual:
                    Remembered saccades

                    Memory:
                    Repetition priming effect
                    Auditory working memory

                    Other:
                    Visual speech perception (mirror neurons?)
                    Attribution of intentions to others
                    Deductive reasoning
                    """,
    'BA19 Left': """
                Secondary visual cortex - Inferior occipital gyrus

                Visual:
                Detection of light intensity
                Detection of patterns
                Tracking visual motion patterns
                Discrimination of finger gestures
                Sustained attention to color and shape
                Feature-based attention
                Orientation-selective attention

                Memory:
                Visual priming
                Visual memory recognition
                Word and face encoding
                Spatial working memory

                Language:
                Processing phonological properties of words (word form?)
                Confrontation naming
                Sign language

                Other:
                Horizontal saccadic eye movements
                Visual mental imagery
                Inferential reasoning
                Visual mental imagery
                """,
    'BA19 Right': """
                Secondary visual cortex - Inferior occipital gyrus
                Visual:
                Detection of light intensity
                Visuo-spatial information processing
                Detection of patterns
                Tracking visual motion patterns
                Discrimination of finger gestures
                Sustained attention to color and shape
                Feature-based attention
                Orientation-selective attention

                Memory:
                Visual priming
                Visual memory recognition
                Word and face encoding
                Spatial working memory

                Language:
                Processing phonological properties of words (word form?)
                Confrontation naming
                Sign language

                Other:
                Face-name association
                Horizontal saccadic eye movements
                Visual mental imagery
                """,
    'BA6 Left': """
                Premotor cortex or Lateral Premotor Area (PMA)
                Also includes Supplementary Motor Area (SMA)

                Motor:
                Motor sequencing/planning
                Motor learning (SMA)
                Movement preparation/imagined movement (Rostral SMA)
                Movement initiation (Caudal SMA)
                Motor imagery (SMA)
                Volitional control of breathing
                Horizontal saccadic eye movements
                Laughter/smiling (SMA)
                Interlimb coordination

                Language:
                Speech motor programming
                Language processing (SMA)
                Language switching
                Reading novel words (aloud and silently)
                Speech perception
                Updating verbal information (Medial)
                Phonological processing
                Object naming
                Lipreading (SMA)
                Word retrieval
                Lexical decision on words and pseudowords
                Syntactical processing

                Memory:
                Working memory
                Mnemonic rehearsal
                Episodic long-term memory
                Topographic memory

                Attention:
                Visuospatial attention
                Visuomotor attention
                Response to visual presentation of letters and pseudoletters
                Updating spatial information (Lateral)
                Visual guided eye movements (frontal eye fields)
                Selective attention to rhythm/processing sequential sounds
                Attention to human voices

                Other:
                Observation of actions (Mirror neurons)
                Planning/solving novel problems
                Executive control of behavior
                Reponse to baroreceptor stimulation
                Generating melodic phrases
                Deductive reasoning
                Formation of qualatative representations
                Processing emotions and self-reflections in decision making
                Calculation
                Temporal context recognition
                Frequency deviant detection
                """,
    'BA6 Right': """ 
                Premotor cortex or Lateral Premotor Area (PMA)
                Also includes Supplementary Motor Area (SMA)

                Motor:
                Motor sequencing/planning
                Motor learning (SMA)
                Movement preparation/imagined movement (Rostral SMA)
                Movement initiation (Caudal SMA)
                Motor imagery (SMA)
                Volitional control of breathing
                Horizontal saccadic eye movements
                Laughter/smiling (SMA)
                Interlimb coordination

                Language:
                Language processing (SMA)
                Language switching
                Speech perception
                Updating verbal information (Medial)
                Lipreading (SMA)
                Word retrieval
                Lexical decision on words and pseudowords
                Syntactical processing

                Memory:
                Working memory
                Mnemonic rehearsal
                Episodic long-term memory
                Topographic memory

                Attention:
                Visuospatial attention
                Visuomotor attention
                Updating spatial information (Lateral)
                Visual guided eye movements (frontal eye fields)
                Attention to human voices

                Other:
                Observation of actions (Mirror neurons)
                Planning/solving novel problems
                Executive control of behavior
                Reponse to baroreceptor stimulation
                Generating melodic phrases
                Response to strong odorant
                Formation of qualatative representations
                Same-different discrimination
                Calculation
                Temporal context recognition
                Frequency deviant detection
                """,
    'BA4 Left': """ 
                Primary motor cortex - Precentral gyrus

                Motor:
                Contralateral finger, hand, and wrist movements (Dorsal)
                Contralateral lip, tongue, face, and mouth movement (Lateral)
                Swallowing / laryngial movement
                Contralateral lower limb (knee, ankle, foot, toe) movement (Mesial)
                Motor imagery
                Learning motor sequences
                Volitional breathing control
                Control of rhythmic motor tasks (i.e. bicycling)
                Inhibition of blinking / voluntary blinking
                Horizontal saccadic eye movements

                Somatosensory:
                Kinesthetic perception of limb movements
                Vibrotactile frequency discrimination
                Finger proprioception
                Thermal hyperalgesia (contralateral)
                Response to touch/observed touch

                Other:
                Attention to action (posterior)
                Topographic memory (motor memory) for visual landmarks
                """,
    'BA4 Right': """ 
                Primary motor cortex - Precentral gyrus

                Motor:
                Contralateral finger, hand, and wrist movements (Dorsal)
                Contralateral lip, tongue, face, and mouth movement (Lateral)
                Swallowing / laryngial movement
                Contralateral lower limb (knee, ankle, foot, toe) movement (Mesial)
                Motor imagery
                Learning motor sequences
                Volitional breathing control
                Control of rhythmic motor tasks (i.e. bicycling)
                Inhibition of blinking / voluntary blinking
                Horizontal saccadic eye movements

                Somatosensory:
                Kinesthetic perception of limb movements
                Vibrotactile frequency discrimination
                Finger proprioception
                Thermal hyperalgesia (contralateral)

                Other:
                Verbal encoding during a non-semantic process
                Attention to action (posterior)
                Topographic memory (motor memory) for visual landmarks
                """,
    'BA7 Left': """ 
                Secondary sensorimotor cortex - Secondary association sensorimotor cortex
                Superior parietal lobule

                Motor:
                Motor imagery
                Processing tool-use gestures
                Motor execution
                Mirror neurons
                Bimanual manipulation
                Saccadic eye movement

                Memory:
                Working memory (motor, visual, auditory, emotional, verbal)
                Conscious recollection of previously experienced events

                Sensory:
                Tactile localization ("where stream")
                Pain perception

                Attention:
                Visuomotor attention

                Language:
                Language processing
                Literal sentence comprehension
                Word comprehension (imageability)
                Attention to phonological relations

                Other:
                Processing emotions and self-reflections during decision making
                Goal-intensive processing
                Temporal context recognition
                """,
    'BA7 Right': """
                Secondary sensorimotor cortex - Secondary association sensorimotor cortex
                Superior parietal lobule

                Visuospatial processing:
                Mental rotation
                Stereopsis
                Perception of personal space
                Line bisection judgments
                Processing chaotic patterns
                Using spatial imagery in deductive reasoning

                Motor:
                Motor imagery
                Motor execution
                Mirror neurons
                Bimanual manipulation
                Saccadic eye movement

                Memory:
                Working memory (motor, visual, auditory, emotional, verbal)
                Visuospatial memory
                Conscious recollection of previously experienced events

                Sensory:
                Tactile localization ("where stream")
                Pain perception

                Attention:
                Visuomotor attention

                Language:
                Language processing
                Literal sentence comprehension
                Word comprehension (imageability)
                Attention to phonological relations

                Other:
                Processing emotions and self-reflections during decision making
                Goal-intensive processing
                """,
}