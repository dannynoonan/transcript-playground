from app.show_metadata import ShowKey
import app.es.es_read_router as esr
from app.nlp.nlp_metadata import MIN_WORDS_FOR_BERT, MAX_WORDS_FOR_BERT, MIN_SPEAKER_LINES, MIN_SPEAKER_LINE_RATIOS


def extract_narrative_sequences(show_key: ShowKey, episode_key: str) -> list:
    '''
    Brute force attempt to extract narrative subplots via speaker co-occurrence    
    '''
    # compile list of episode speakers sorted desc by line count
    scene_events_by_speaker_response = esr.agg_scene_events_by_speaker(show_key, episode_key=episode_key)
    speaker_line_counts = scene_events_by_speaker_response['scene_events_by_speaker']
    del speaker_line_counts['_ALL_']
    sorted_speakers = [spkr for spkr, ct in speaker_line_counts.items() if ct > MIN_SPEAKER_LINES]

    # combine speakers into groups of 2, 3, and 4, sorted desc by group size and line count 
    speaker_duos = []
    for i in range(len(sorted_speakers)):
        for j in range(i+1, len(sorted_speakers)):
            speaker_duos.append([sorted_speakers[i], sorted_speakers[j]])
    speaker_trios = []
    for i in range(len(sorted_speakers)):
        for j in range(i+1, len(sorted_speakers)):
            for k in range(j+1, len(sorted_speakers)):
                speaker_trios.append([sorted_speakers[i], sorted_speakers[j], sorted_speakers[k]])
    speaker_quads = []
    for i in range(len(sorted_speakers)):
        for j in range(i+1, len(sorted_speakers)):
            for k in range(j+1, len(sorted_speakers)):
                for l in range(k+1, len(sorted_speakers)):
                    speaker_quads.append([sorted_speakers[i], sorted_speakers[j], sorted_speakers[k], sorted_speakers[l]])

    speaker_groups = speaker_quads
    speaker_groups.extend(speaker_trios)
    speaker_groups.extend(speaker_duos)

    # initialize narrative_sequences we'll be compiling as output, also track which scenes we're ultimately sourcing from
    narrative_sequences = []
    # all_sourced_scene_wcs = {}

    for speaker_group in speaker_groups:
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print(f'BEGIN `search_scene_events_multi_speaker` for speaker_group={speaker_group}')
        search_response = esr.search_scene_events_multi_speaker(show_key, ','.join(speaker_group), episode_key=episode_key, intersection=True)
        if 'scene_count' not in search_response or search_response['scene_count'] < 2:
            # print(f'fewer than 2 scenes match speaker_group={speaker_group}, skipping')
            continue

        speaker_line_counts = {s:0 for s in speaker_group}
        narrative_lines = []
        source_scene_wcs = {}
        for episode in search_response['matches']:
            for scene in episode['scenes']:
                scene_used = False
                # lines_used = 0
                wc = 0
                for scene_event in scene['scene_events']:
                    use_line = True
                    for ns in narrative_sequences:
                        if scene_event['spoken_by'] in ns['speaker_group'] and scene['sequence'] in ns['source_scene_wcs']:
                            use_line = False
                            break
                    if use_line:
                        narrative_lines.append(scene_event['dialog'])
                        speaker_line_counts[scene_event['spoken_by']] += 1
                        # lines_used += 1
                        wc += len(scene_event['dialog'].split(' '))
                        scene_used = True
                if scene_used:
                    if scene['sequence'] not in source_scene_wcs:
                        source_scene_wcs[scene['sequence']] = 0
                    source_scene_wcs[scene['sequence']] += wc
        if len(narrative_lines) < MIN_SPEAKER_LINES:
            print(f'{len(narrative_lines)} lines exchanged within speaker_group={speaker_group} does not meet threshold of {MIN_SPEAKER_LINES}, skipping')
            continue

        valid_speaker_group = True
        total_lines = sum(speaker_line_counts.values())
        for spkr, line_ct in speaker_line_counts.items():
            line_pct = line_ct / total_lines
            if line_ct < MIN_SPEAKER_LINES or line_pct < MIN_SPEAKER_LINE_RATIOS[len(speaker_group)]:
                # print('========================================================================================')
                # print(f'line_ct={line_ct} for spkr={spkr} within speaker_line_counts={speaker_line_counts} does not meet min_speaker_lines threshold of {MIN_SPEAKER_LINES}, skipping')
                # print(f'line_pct={line_pct} for spkr={spkr} within speaker_line_counts={speaker_line_counts} does not meet min_speaker_line_ratios threshold of {MIN_SPEAKER_LINE_RATIOS[len(speaker_group)]}, skipping')
                valid_speaker_group = False
                continue
            # if line_ct < MIN_SPEAKER_LINES:
            #     # print('----------------------------------------------------------------------------------------')
            #     valid_speaker_group = False
            #     continue

        if valid_speaker_group:
            narrative_text = ' '.join(narrative_lines)
            # wc = len(narrative_text.split(' '))
            # narrative_sequences.append(dict(speaker_group=speaker_group, narrative_text=narrative_text, wc=wc, source_scene_wcs=source_scene_wcs, speaker_line_counts=speaker_line_counts))
            narrative_sequences.append(dict(speaker_group=speaker_group, narrative_lines=narrative_lines, wc=len(narrative_text.split(' ')), 
                                            source_scene_wcs=source_scene_wcs, speaker_line_counts=speaker_line_counts))



            # if wc < MAX_WORDS_FOR_BERT:
            #     narrative_sequences.append(dict(speaker_group=speaker_group, narrative_text=narrative_text, wc=wc,
            #                                     source_scene_wcs=source_scene_wcs, speaker_line_counts=speaker_line_counts))
            # else:
            #     # print(f'{len(narrative_text)} words exchanged within speaker_group={speaker_group} exceeds threshold of {MAX_WORDS_FOR_BERT}, breaking up')
            #     wc = 0
            #     narrative_subseq = []
            #     for i, scene_event in enumerate(narrative_lines):
            #         line_wc = len(scene_event.split(' '))
            #         if (wc + line_wc) > MAX_WORDS_FOR_BERT:
            #             narrative_text = ' '.join(narrative_subseq)
            #             narrative_sequences.append(dict(speaker_group=speaker_group, narrative_text=narrative_text, wc=wc,
            #                                             source_scene_wcs=source_scene_wcs, speaker_line_counts=speaker_line_counts))
            #             narrative_subseq = []
            #             wc = 0
            #         narrative_subseq.append(scene_event)
            #         wc += line_wc
            #         if i == len(narrative_lines)-1:
            #             narrative_text = ' '.join(narrative_subseq)
            #             if len(narrative_text.split(' ')) > MIN_WORDS_FOR_BERT:
            #                 narrative_sequences.append(dict(speaker_group=speaker_group, narrative_text=narrative_text, wc=wc,
            #                                                 source_scene_wcs=source_scene_wcs, speaker_line_counts=speaker_line_counts))
            # for ss, wc in source_scene_wcs.items():
            #     if ss not in all_sourced_scene_wcs:
            #         all_sourced_scene_wcs[ss] = 0
            #     all_sourced_scene_wcs[ss] += wc

    return narrative_sequences
