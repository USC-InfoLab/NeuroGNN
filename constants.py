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


# Regions of the brain
# CORTEX_REGIONS = [
#     'Frontal Lobe',
#     'Parietal Lobe',
#     'Left Temporal Lobe',
#     'Right Temporal Lobe',
#     'Occipital Lobe',
#     'Central Gyrus'
# ]
   
CORTEX_REGIONS = [
    'Pre-Frontal Lobe',
    'Frontal Lobe',
    'Parietal Lobe',
    'Temporal Lobe',
    'Occipital Lobe',
    'Central Gyrus'
]

# Dictionary describing eeg electrodes
# ELECTRODES_DESCRIPTIONS = {
#     'EEG FP1': 'Fp1 is an EEG electrode positioned at the frontopolar region of the scalp on the left side. It is commonly used to record electrical brain activity in that specific area. The Fp1 electrode is important for capturing frontal lobe activity, including cognitive processes such as attention, decision-making, and emotional responses. It plays a crucial role in EEG monitoring and can provide valuable insights into brain function.',
#     'EEG FP2': 'Fp2 is an EEG electrode placed at the frontopolar region of the scalp on the right side. It is similar to Fp1 in terms of function and purpose. By recording electrical signals from the right frontopolar area, the Fp2 electrode helps monitor activity in the frontal lobe. This electrode can be instrumental in detecting abnormalities or changes in cognitive processing, emotional regulation, and other functions associated with the frontal brain regions.',
#     'EEG F3': 'F3 is an EEG electrode positioned on the left side of the scalp, over the frontal lobe. It captures electrical brain activity from the left frontal region and plays a crucial role in monitoring cognitive processes, attention, and motor planning associated with the left hemisphere. The F3 electrode is essential for evaluating frontal lobe abnormalities and can provide valuable insights into conditions such as epilepsy, ADHD, and executive function disorders.',
#     'EEG F4': "F4 is an EEG electrode situated on the right side of the scalp, mirroring F3's position. It records electrical signals originating from the right frontal lobe. Similar to F3, the F4 electrode is vital for assessing cognitive functions, attention, and motor planning associated with the right hemisphere. Monitoring the electrical activity in this area is crucial for detecting abnormalities or changes in brain function and can aid in the diagnosis and management of various neurological disorders.",
#     'EEG C3': "C3 is an EEG electrode placed on the left side of the scalp, over the Central Gyrus. It captures electrical brain activity from the left central area, including the sensorimotor cortex. The C3 electrode is important for studying motor control, movement planning, and somatosensory processing associated with the left hemisphere. Monitoring this region can help identify abnormalities or disruptions in motor function and contribute to the evaluation of conditions such as stroke, movement disorders, and brain injuries.",
#     'EEG C4': "C4 is an EEG electrode located on the right side of the scalp, corresponding to C3's position. It records electrical signals from the right Central Gyrus, encompassing the sensorimotor cortex. Similar to C3, the C4 electrode is crucial for monitoring motor control, movement planning, and somatosensory processing associated with the right hemisphere. It plays a significant role in assessing motor function asymmetries and can aid in the diagnosis and treatment of conditions such as Parkinson's disease, motor cortex lesions, and focal seizures.",
#     'EEG P3': "P3 is an EEG electrode positioned on the left side of the scalp, above the parietal lobe. It captures electrical brain activity from the left parietal region, which is involved in processes such as spatial awareness, attention, and sensory integration. The P3 electrode is essential for studying visuospatial processing, visual attention, and other functions associated with the left hemisphere's parietal areas. Monitoring this region can provide valuable insights into conditions like neglect syndrome, spatial processing disorders, and attentional deficits.",
#     'EEG P4': "P4 is an EEG electrode situated on the right side of the scalp, mirroring P3's position. It records electrical signals from the right parietal lobe. The P4 electrode is instrumental in monitoring visuospatial processing, attention, and sensory integration associated with the right hemisphere's parietal regions. By assessing electrical activity in this area, it can help identify abnormalities or changes in brain function and contribute to the evaluation and management of conditions such as spatial neglect, visual attention disorders, and parietal lobe epilepsy.",
#     'EEG O1': "O1 is an EEG electrode positioned on the left side of the scalp, over the occipital lobe. It captures electrical brain activity from the left occipital region, which is primarily responsible for visual processing and perception. The O1 electrode is essential for studying visual evoked potentials, visual attention, and other functions associated with the left hemisphere's occipital areas. Monitoring this region can provide valuable insights into conditions such as visual processing disorders, occipital lobe epilepsy, and visual hallucinations.",
#     'EEG O2': "O2 is an EEG electrode situated on the right side of the scalp, corresponding to O1's position. It records electrical signals from the right occipital lobe. Similar to O1, the O2 electrode is vital for monitoring visual processing, visual attention, and perception associated with the right hemisphere's occipital areas. It plays a significant role in assessing visual function asymmetries and can aid in the diagnosis and management of conditions such as visual field defects, occipital seizures, and visual processing impairments.",
#     'EEG F7': "F7 is an EEG electrode situated at the left frontotemporal region of the scalp. It captures electrical activity from the left side of the brain, specifically the frontal and temporal lobes. The F7 electrode plays a significant role in assessing brain functions related to language processing, memory, and emotion. It is particularly useful for investigating disorders like epilepsy and monitoring the presence of abnormal electrical patterns in these areas.",
#     'EEG F8': "F8 is an EEG electrode positioned at the right frontotemporal region of the scalp. It complements F7 by capturing electrical brain activity from the right frontal and temporal lobes. The F8 electrode helps monitor cognitive functions associated with the right hemisphere, including language processing, memory retrieval, and emotional regulation. It is essential for identifying any asymmetries or abnormalities in these brain regions and can contribute to the diagnosis and treatment of various neurological conditions.",
#     'EEG T3': "T3 is an EEG electrode located on the left side of the scalp, above the temporal lobe. It records electrical signals originating from the left temporal region of the brain. The T3 electrode is significant for monitoring auditory processing, language comprehension, and memory functions associated with the left hemisphere. It is commonly used in diagnosing and studying conditions such as temporal lobe epilepsy and language-related disorders.",
#     'EEG T4': "T4 is an EEG electrode placed on the right side of the scalp, above the temporal lobe. It complements T3 by recording electrical brain activity from the right temporal region. By monitoring the right hemisphere's functions related to auditory processing, language comprehension, and memory, the T4 electrode assists in assessing brain activity asymmetries and identifying abnormalities in these areas. It is particularly useful in the evaluation of temporal lobe epilepsy and language disorders.",
#     'EEG T5': "T5 is an EEG electrode placed on the left side of the scalp, above the temporal lobe, but slightly posterior to T3. It captures electrical brain activity from the left temporal region, particularly the superior and posterior aspects. The T5 electrode is important for studying auditory processing, language comprehension, and memory functions associated with the left hemisphere's superior temporal gyrus. Monitoring this region can provide valuable insights into conditions such as temporal lobe epilepsy, auditory processing disorders, and language impairments.",
#     'EEG T6': "T6 is an EEG electrode located on the right side of the scalp, corresponding to T5's position. It records electrical signals from the right temporal region, particularly the superior and posterior aspects of the superior temporal gyrus. Similar to T5, the T6 electrode is crucial for monitoring auditory processing, language comprehension, and memory functions associated with the right hemisphere. It plays a significant role in assessing auditory function asymmetries and can aid in the diagnosis and treatment of conditions such as temporal lobe epilepsy, auditory hallucinations, and language disorders.",
#     'EEG FZ': "FZ is an EEG electrode positioned at the midline of the scalp, between F3 and F4. It captures electrical brain activity from the frontal-Central Gyrus known as the midline prefrontal cortex. The FZ electrode is important for studying cognitive processes, working memory, and attention regulation. Monitoring this midline region can provide valuable insights into executive functions, emotional regulation, and frontal lobe abnormalities. It is particularly useful in the evaluation of conditions such as attention deficit hyperactivity disorder (ADHD), frontal lobe epilepsy, and mood disorders.",
#     'EEG CZ': "CZ is an EEG electrode placed at the midline of the scalp, between C3 and C4. It records electrical signals from the central-parietal region, encompassing the sensorimotor and somatosensory cortices. The CZ electrode is crucial for monitoring motor control, sensory integration, and somatosensory processing. It plays a significant role in assessing abnormalities or changes in sensorimotor functions and can contribute to the evaluation and treatment of conditions such as movement disorders, sensory processing disorders, and central seizures.",
#     'EEG PZ': "PZ is an EEG electrode positioned at the midline of the scalp, between P3 and P4. It captures electrical brain activity from the parietal-occipital region, which encompasses the parietal lobes and the posterior aspects of the occipital lobes. The PZ electrode plays a crucial role in studying visuospatial processing, attention, and sensory integration in the parietal and occipital regions. It is particularly useful for monitoring visual-spatial cognition, visual attention, and multisensory integration processes. The PZ electrode can provide valuable insights into conditions such as spatial neglect, visuospatial processing disorders, and parietal lobe epilepsy. By monitoring electrical signals from this region, it contributes to the evaluation and understanding of brain activity patterns related to visuospatial perception and attentional processes."
# }

# ELECTRODES_DESCRIPTIONS = {
#     'EEG FP1': 'frontopolar, left side, cognitive processes, attention, decision-making',
#     'EEG FP2': 'frontopolar, right side, cognitive processes, emotional regulation',
#     'EEG F3': 'left side, frontal lobe, cognitive processes, attention, motor planning',
#     'EEG F4': 'right side, frontal lobe, cognitive functions, attention, motor planning',
#     'EEG C3': 'left side, Central Gyrus, motor control, movement planning, somatosensory processing',
#     'EEG C4': 'right side, Central Gyrus, motor control, movement planning, somatosensory processing',
#     'EEG P3': 'left side, parietal lobe, spatial awareness, attention, sensory integration',
#     'EEG P4': 'right side, parietal lobe, visuospatial processing, attention, sensory integration',
#     'EEG O1': 'left side, occipital lobe, visual processing, perception',
#     'EEG O2': 'right side, occipital lobe, visual processing, perception',
#     'EEG F7': 'left frontotemporal, language processing, memory, emotion',
#     'EEG F8': 'right frontotemporal, language processing, memory, emotional regulation',
#     'EEG T3': 'left side, temporal lobe, auditory processing, language comprehension, memory',
#     'EEG T4': 'right side, temporal lobe, auditory processing, language comprehension, memory',
#     'EEG T5': 'left side, temporal lobe, auditory processing, language comprehension, memory',
#     'EEG T6': 'right side, temporal lobe, auditory processing, language comprehension, memory',
#     'EEG FZ': 'midline, frontal-Central Gyrus, cognitive processes, working memory, attention',
#     'EEG CZ': 'midline, central-parietal region, motor control, sensory integration, somatosensory processing',
#     'EEG PZ': 'midline, parietal-occipital region, visuospatial processing, attention, sensory integration'
# }

# ELECTRODES_BROADMANN_MAPPING = {
#     'EEG FP1': [10, 11, 46],
#     'EEG FP2': [10, 11, 46],
#     'EEG F3': [8, 9, 46],
#     'EEG F4': [8, 9, 46],
#     'EEG C3': [3, 1, 4],
#     'EEG C4': [3, 1, 4],
#     'EEG P3': [7, 40, 19],
#     'EEG P4': [7, 40, 19],
#     'EEG O1': [18, 19, 17],
#     'EEG O2': [18, 19, 17],
#     'EEG F7': [45, 47, 46],
#     'EEG F8': [45, 47, 46],
#     'EEG T3': [42, 22, 21],
#     'EEG T4': [42, 22, 21],
#     'EEG T5': [39, 37, 19],
#     'EEG T6': [39, 37, 19],
#     'EEG FZ': [8, 6, 9],
#     'EEG CZ': [6, 4, 3],
#     'EEG PZ': [7, 5, 19]
# }


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


# Dictionary describing brain regions
# CORTEX_REGIONS_DESCRIPTIONS = {
#     'Frontal Lobe': 'The frontal lobe, located at the front of the brain, is involved in a wide range of higher cognitive functions. It plays a crucial role in executive functions such as decision-making, problem-solving, planning, and reasoning. Additionally, the frontal lobe contributes to motor control, including the initiation and coordination of voluntary movements. It also influences personality, social behavior, emotional regulation, and attentional processes. EEG electrodes associated with the frontal lobe include Fp1, Fp2, F3, F4, F7, F8, and FZ.',
#     'Parietal Lobe': 'The parietal lobe, situated near the top and back of the brain, is responsible for various functions related to sensory perception and spatial awareness. It integrates sensory information from different modalities, such as touch, temperature, and proprioception, to create our perception of the surrounding world. The parietal lobe is also involved in spatial cognition, attentional processes, and the coordination of movements. It helps us navigate our environment and manipulate objects in space. EEG electrodes associated with the parietal lobe include P3, P4, PZ, and sometimes T5 and T6.',
#     'Left Temporal Lobe': "The left temporal lobe, located on the left side of the brain, is involved in several crucial functions. It plays a critical role in auditory processing, allowing us to perceive and interpret sounds. In particular, the left temporal lobe is involved in language comprehension and production, including the understanding and generation of spoken and written language. It houses important language-related structures, such as Wernicke's area, which contributes to language processing and semantic understanding. Additionally, the left temporal lobe is involved in memory formation, including the encoding and retrieval of verbal and auditory information. EEG electrodes associated with the left temporal lobe include T3, T5, and sometimes P3.",
#     'Right Temporal Lobe': "The right temporal lobe, positioned on the right side of the brain, shares several functions with the left temporal lobe. It is involved in auditory processing, including the perception and interpretation of sounds. While the left temporal lobe primarily handles language functions, the right temporal lobe is crucial for the processing of non-verbal auditory information, such as music and environmental sounds. It also contributes to aspects of visual-spatial processing and facial recognition. Additionally, the right temporal lobe plays a role in memory formation, particularly in the retrieval of non-verbal and visuospatial memories. EEG electrodes associated with the right temporal lobe include T4, T6, and sometimes P4.",
#     'Occipital Lobe': "The occipital lobe, located at the back of the brain, is primarily dedicated to visual processing. It receives and processes visual information from the eyes, allowing us to perceive and interpret the world around us. The occipital lobe contains specialized regions that process different aspects of vision, such as object recognition, color perception, and motion detection. It helps us form visual representations of the environment and allows us to recognize and identify objects, faces, and visual patterns. EEG electrodes associated with the occipital lobe include O1 and O2.",
#     'Central Gyrus': "The Central Gyrus encompasses the sensorimotor cortex, which is responsible for motor control and somatosensory processing. It plays a critical role in planning and executing voluntary movements. The Central Gyrus receives sensory information related to touch, pressure, pain, and temperature, providing us with a sense of our body's position, movement, and interaction with the environment. This region is involved in the coordination and modulation of movements, integrating sensory feedback with motor commands to ensure smooth and precise execution of actions. The Central Gyrus helps us manipulate objects, perform complex motor tasks, and engage in activities requiring fine motor control. EEG electrodes associated with the Central Gyrus include C3, C4, CZ, and sometimes F3 and F4."
# }

# CORTEX_REGIONS_DESCRIPTIONS = {
#     'Frontal Lobe': 'Higher cognitive functions, executive functions, motor control, personality',
#     'Parietal Lobe': 'Sensory perception, spatial awareness, spatial cognition, movement coordination',
#     'Left Temporal Lobe': 'Auditory processing, language comprehension, memory formation',
#     'Right Temporal Lobe': 'Auditory processing, non-verbal auditory information, visual-spatial processing',
#     'Occipital Lobe': 'Visual processing, object recognition, color perception, motion detection',
#     'Central Gyrus': 'Motor control, somatosensory processing, motor coordination, fine motor control',
# }

# CORTEX_REGIONS_DESCRIPTIONS = {
#     'Frontal Lobe': 'Higher cognitive functions, executive functions, motor control, personality',
#     'Parietal Lobe': 'Sensory perception, spatial awareness, spatial cognition, movement coordination',
#     'Temporal Lobe': 'Auditory processing, language comprehension, non-verbal auditory information, visual-spatial processing, and memory formation',
#     'Occipital Lobe': 'Visual processing, object recognition, color perception, motion detection',
#     'Central Gyrus': 'Motor control, somatosensory processing, motor coordination, fine motor control',
# }


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


# Dictionary mapping brain regions to eeg electrodes
# ELECTRODES_REGIONS = {
#     'EEG FP1': 'Frontal Lobe',
#     'EEG FP2': 'Frontal Lobe',
#     'EEG F3': 'Frontal Lobe',
#     'EEG F4': 'Frontal Lobe',
#     'EEG C3': 'Central Gyrus',
#     'EEG C4': 'Central Gyrus',
#     'EEG P3': 'Parietal Lobe',
#     'EEG P4': 'Parietal Lobe',
#     'EEG O1': 'Occipital Lobe',
#     'EEG O2': 'Occipital Lobe',
#     'EEG F7': 'Frontal Lobe',
#     'EEG F8': 'Frontal Lobe',
#     'EEG T3': 'Left Temporal Lobe',
#     'EEG T4': 'Right Temporal Lobe',
#     'EEG T5': 'Left Temporal Lobe',
#     'EEG T6': 'Right Temporal Lobe',
#     'EEG FZ': 'Frontal Lobe',
#     'EEG CZ': 'Central Gyrus',
#     'EEG PZ': 'Parietal Lobe'
# }

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