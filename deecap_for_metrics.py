import torch
import torch.nn as nn
import torch.nn.functional as F


class ImitationNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        return self.mlp(x)


class MapImitationCNN(nn.Module):
    def __init__(self, num_queries=200, hidden_dim=512):
        super().__init__()

        # 1x1 Convolution to reduce channel size (queries as channels)
        self.conv1 = nn.Conv2d(num_queries, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, num_queries, kernel_size=1)  # Restore queries as channels

        # Adaptive pooling to handle variable spatial sizes
        # self.adaptive_pool = nn.AdaptiveAvgPool2d(self.output_hw)

    def forward(self, x, h, w):
        batch_size, queries, hw = x.shape

        x = x.view(batch_size, queries, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        # Flatten back
        x = x.view(batch_size, queries, hw)
        return x


from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

class DeecapRelTR(nn.Module):
    def __init__(self, reltr_model):
        super().__init__()
        self.reltr = reltr_model
        num_layers = reltr_model.transformer.decoder.num_layers
        # so_feature_dim = reltr_model.so_mask_fc.out_features
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.imitation_networks_enitity = nn.ModuleList([
              ImitationNetwork(reltr_model.transformer.d_model).to(device) for _ in range(num_layers)
          ])

        self.imitation_networks_subject = nn.ModuleList([
              ImitationNetwork(reltr_model.transformer.d_model).to(device) for _ in range(num_layers)
          ])

        self.imitation_networks_object = nn.ModuleList([
              ImitationNetwork(reltr_model.transformer.d_model).to(device) for _ in range(num_layers)
          ])

        self.imitation_networks_submap = nn.ModuleList([
              MapImitationCNN().to(device) for _ in range(num_layers)
          ])

        self.imitation_networks_objmap = nn.ModuleList([
              MapImitationCNN().to(device) for _ in range(num_layers)
          ])

        # print("hidden dim", reltr_model.transformer.d_model)

        self.num_layers = num_layers
        self.return_intermediate = True
        self.layer_outputs = []


    def forward(self, samples, confidence_threshold=0, verbose=False, layer_exit = -1, imitate=True):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # run backbone
        features, pos = self.reltr.backbone(samples)
        src_raw, mask = features[-1].decompose()
        assert mask is not None

        #encoder param mapping
        entity_embed = self.reltr.entity_embed.weight
        triplet_embed = self.reltr.triplet_embed.weight
        pos_embed = pos[-1]
        so_embed = self.reltr.so_embed.weight
        src = self.reltr.input_proj(src_raw)

        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # print(len(torch.split(entity_embed, c, dim=1)))
        entity_embed, entity = torch.split(entity_embed, c, dim=1)
        triplet_embed, triplet = torch.split(triplet_embed, [c, 2 * c], dim=1)

        entity_embed = entity_embed.unsqueeze(1).repeat(1, bs, 1)
        triplet_embed = triplet_embed.unsqueeze(1).repeat(1, bs, 1)
        entity = entity.unsqueeze(1).repeat(1, bs, 1)
        triplet = triplet.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        memory = self.reltr.transformer.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        self.memory = memory # save for imitation loss

        # decoder param mapping
        memory_key_padding_mask=mask
        tgt_mask = None
        memory_mask = None
        tgt_key_padding_mask = None
        pos=pos_embed
        entity_pos=entity_embed
        triplet_pos=triplet_embed
        so_pos=so_embed

        # self.reltr.transformer.decoder(entity, triplet, memory, memory_key_padding_mask=mask,
        #                                             pos=pos_embed, entity_pos=entity_embed,
        #                                             triplet_pos=triplet_embed, so_pos=so_embed)

        output_entity = entity
        output_triplet = triplet
        intermediate_entity = []
        intermediate_triplet = []
        intermediate_submaps = []
        intermediate_objmaps = []
        batch_exit_layers = []

        for i,layer in enumerate(self.reltr.transformer.decoder.layers):
            output_entity, output_triplet, sub_maps, obj_maps = layer(output_entity, output_triplet, entity_pos, triplet_pos, so_pos,
                                                                      memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                                                      memory_key_padding_mask=memory_key_padding_mask, pos=pos)


            # print("single layer triplet", output_triplet.shape)
            if self.return_intermediate:
                intermediate_entity.append(output_entity)
                intermediate_triplet.append(output_triplet)
                intermediate_submaps.append(sub_maps)
                intermediate_objmaps.append(obj_maps)

            """
            Current flow takes the mth layer representation, calculates logits for object and subject class. If the prob distributions have low entropy,
            ie, high confidence, it uses the imitation layer representations of [m+1, n]th layers and then makes a final prediction.

            Calculate/store the layer of exit.
            """
            # print("Real decoder executed for layer", i)
            if not self.training:
                # Compute confidence score for current layer

                # hs = output_entity
                # hs_sub, hs_obj = torch.split(output_triplet, self.reltr.hidden_dim, dim=-1) #subject and object hidden states for decoder layer m
                # print("intermediate shapes", hs_sub.shape, hs_obj.shape)
                hs = torch.stack(intermediate_entity)
                hs_t = torch.stack(intermediate_triplet)
                
                hs = hs.transpose(1, 2)
                hs_t = hs_t.transpose(1, 2)
                hs_sub, hs_obj = torch.split(hs_t, self.reltr.hidden_dim, dim=-1)

                outputs_class = self.reltr.entity_class_embed(hs)
                outputs_coord = self.reltr.entity_bbox_embed(hs).sigmoid()

                outputs_class_sub = self.reltr.sub_class_embed(hs_sub) #get subject class logits from internal representation of layer m
                outputs_coord_sub = self.reltr.sub_bbox_embed(hs_sub).sigmoid() #get subject class bbox
                # print("output class shape for sub", outputs_class_sub.shape)

                # print("sub_map size, obj_map size", intermediate_submaps_early.shape, intermediate_objmaps_early.shape)
                # print("output class logits", outputs_class_sub[0])

                outputs_class_obj = self.reltr.obj_class_embed(hs_obj) #get object class logits from internal representation of layer m
                outputs_coord_obj = self.reltr.obj_bbox_embed(hs_obj).sigmoid() #get object class bbox
                # print("output class shape for obj", outputs_class_obj.shape)

                entropy = self.compute_confidence(outputs_class, outputs_class_sub, outputs_class_obj, i) #compute entropy over num_class probabilities for layer m output
                if verbose:
                    print('layer:', i)
                    print("entropy", entropy)
                if entropy <= confidence_threshold or i==layer_exit:
                    # Early exit - imitate deeper layers
                    # final_entity = hs
                    # final_subject_hs = hs_sub
                    # final_object_hs = hs_obj
                    # final_submap_hs = sub_maps
                    # final_objmap_hs = obj_maps
                    # print(final_entity.shape, final_subject_hs.shape, final_object_hs.shape, final_submap_hs.shape, final_objmap_hs.shape)
                    batch_exit_layers.append(i)

                    final_entity = torch.mean(torch.stack(intermediate_entity), dim=0).transpose(0,1)
                    temp = torch.mean(torch.stack(intermediate_triplet), dim=0).transpose(0,1)
                    final_subject_hs,final_object_hs = torch.split(temp, self.reltr.hidden_dim, dim=-1)
                    final_submap_hs = torch.mean(torch.stack(intermediate_submaps), dim=0)
                    final_objmap_hs = torch.mean(torch.stack(intermediate_objmaps), dim=0)
                    # print(final_entity.shape, final_subject_hs.shape, final_object_hs.shape, final_submap_hs.shape, final_objmap_hs.shape)


                    intermediate_entity_early = intermediate_entity.copy()
                    intermediate_triplet_early = intermediate_triplet.copy()
                    intermediate_submaps_early = intermediate_submaps.copy()
                    intermediate_objmaps_early = intermediate_objmaps.copy()


                    # Apply imitation networks to approximate deeper layers
                    if imitate:
                      for k in range(i + 1, self.num_layers):
                        # Currently, this doesn't use h_shallow or any feature fusion strategy
                        final_entity = self.imitation_networks_enitity[k](final_entity)
                        final_subject_hs = self.imitation_networks_subject[k](final_subject_hs)
                        final_object_hs = self.imitation_networks_object[k](final_object_hs)
                        # final_submap_hs = self.imitation_networks_submap[k](final_submap_hs)
                        # final_objmap_hs = self.imitation_networks_objmap[k](final_objmap_hs)

                        final_submap_hs = self.imitation_networks_submap[k](final_submap_hs,h,w)
                        final_objmap_hs = self.imitation_networks_objmap[k](final_objmap_hs,h,w)
                        # print("asdf", final_objmap_hs.shape, final_submap_hs.shape)

                        intermediate_entity_early.append(final_entity.transpose(0,1))
                        intermediate_triplet_early.append(torch.cat((final_subject_hs.transpose(0,1), final_object_hs.transpose(0,1)), dim=-1))
                        intermediate_submaps_early.append(final_submap_hs)
                        intermediate_objmaps_early.append(final_objmap_hs)


                    hs = torch.stack(intermediate_entity_early)
                    hs_t = torch.stack(intermediate_triplet_early)
                    sub_maps = torch.stack(intermediate_submaps_early)
                    obj_maps = torch.stack(intermediate_objmaps_early)

                    so_masks = torch.cat((sub_maps.reshape(sub_maps.shape[0], bs, sub_maps.shape[2], 1, h, w),
                              obj_maps.reshape(obj_maps.shape[0], bs, obj_maps.shape[2], 1, h, w)), dim=3) #decoder + imitation stack
                    hs = hs.transpose(1, 2)
                    hs_t = hs_t.transpose(1, 2)
                    memory_permute = memory.permute(1, 2, 0).view(bs, c, h, w)
                    so_masks = so_masks.detach()
                    so_masks = self.reltr.so_mask_conv(so_masks.view(-1, 2, src_raw.shape[-2],src_raw.shape[-1])).view(hs_t.shape[0], hs_t.shape[1], hs_t.shape[2],-1)
                    so_masks = self.reltr.so_mask_fc(so_masks)

                    hs_sub, hs_obj = torch.split(hs_t, self.reltr.hidden_dim, dim=-1)

                    outputs_class = self.reltr.entity_class_embed(hs)
                    outputs_coord = self.reltr.entity_bbox_embed(hs).sigmoid()

                    outputs_class_sub = self.reltr.sub_class_embed(hs_sub)
                    outputs_coord_sub = self.reltr.sub_bbox_embed(hs_sub).sigmoid()
                    # print("output class shape for sub", outputs_class_sub.shape)

                    outputs_class_obj = self.reltr.obj_class_embed(hs_obj)
                    outputs_coord_obj = self.reltr.obj_bbox_embed(hs_obj).sigmoid()
                    # print("output class shape for obj", outputs_class_obj.shape)

                    outputs_class_rel = self.reltr.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1))
                    out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
                            'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
                            'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
                            'rel_logits': outputs_class_rel[-1]}
                    if verbose:
                      print("early exit after layer", i)
                    if self.reltr.aux_loss:
                      out['aux_outputs'] = self.reltr._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                    outputs_class_obj, outputs_coord_obj, outputs_class_rel)
                    return out, entropy, batch_exit_layers

        #TODO: training and imitation loss functions
        hs = torch.stack(intermediate_entity)
        hs_t = torch.stack(intermediate_triplet)
        sub_maps = torch.stack(intermediate_submaps)
        obj_maps = torch.stack(intermediate_objmaps)
        # print("sub_maps, obj_maps shape", sub_maps.shape, obj_maps.shape)

        # self.entity_outputs = hs.transpose(1, 2)
        # self.triplet_outputs = hs_t.transpose(1, 2)
        self.entity_outputs = hs
        self.triplet_outputs = hs_t
        self.submap_outputs = sub_maps
        self.objmap_outputs = obj_maps
        self.h = h
        self.w = w
        self.bs = bs
        self.c = c
        self.src_raw_shape = src_raw.shape

        #num_layers, bs, num_queries,1,h,w
        so_masks = torch.cat((sub_maps.reshape(sub_maps.shape[0], bs, sub_maps.shape[2], 1, h, w),
                              obj_maps.reshape(obj_maps.shape[0], bs, obj_maps.shape[2], 1, h, w)), dim=3)
        # print("hs, hs_t before", hs.shape, hs_t.shape)
        hs = hs.transpose(1, 2)
        hs_t = hs_t.transpose(1, 2)
        # print("hs, hs_t", hs.shape, hs_t.shape)
        memory_permute = memory.permute(1, 2, 0).view(bs, c, h, w)

        # print("shape test", hs.shape)
        # print(hs_t.shape)

        so_masks = so_masks.detach()
        # print("so shape", so_masks.shape)
        so_masks = self.reltr.so_mask_conv(so_masks.view(-1, 2, src_raw.shape[-2],src_raw.shape[-1])).view(hs_t.shape[0], hs_t.shape[1], hs_t.shape[2],-1)
        so_masks = self.reltr.so_mask_fc(so_masks)

        hs_sub, hs_obj = torch.split(hs_t, self.reltr.hidden_dim, dim=-1)

        self.sub_outputs = hs_sub
        self.obj_outputs = hs_obj
        self.so_fc_outputs = so_masks

        # print("intermediate shapes", hs_sub.shape, hs_obj.shape)
        outputs_class = self.reltr.entity_class_embed(hs)
        outputs_coord = self.reltr.entity_bbox_embed(hs).sigmoid()

        outputs_class_sub = self.reltr.sub_class_embed(hs_sub)
        outputs_coord_sub = self.reltr.sub_bbox_embed(hs_sub).sigmoid()
        # print("output class shape for sub", outputs_class_sub.shape)

        outputs_class_obj = self.reltr.obj_class_embed(hs_obj)
        outputs_coord_obj = self.reltr.obj_bbox_embed(hs_obj).sigmoid()
        # print("output class shape for obj", outputs_class_obj.shape)

        outputs_class_rel = self.reltr.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1))
        # print("output class shape for rel", outputs_class_rel.shape)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
               'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
               'rel_logits': outputs_class_rel[-1]}
        if self.reltr.aux_loss:
            out['aux_outputs'] = self.reltr._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                    outputs_class_obj, outputs_coord_obj, outputs_class_rel)
        # print("all layers executed")
        return out, entropy, batch_exit_layers
        # return self.reltr.predict_triplets(hs_sub, hs_obj), self.num_layers - 1


    def compute_confidence(self, entity_queries, subject_queries, object_queries, layer_idx):
      combined = torch.cat([subject_queries, object_queries], dim=0)
      prob = torch.nn.functional.softmax(combined, dim=-1)
      return -torch.sum(prob * torch.log(prob), dim=-1).mean()


    def compute_imitation_loss(self, criterion, targets):
        imitation_loss_total = 0.0
        task_loss_total = 0.0

        intermediate_entity = []
        intermediate_triplet = []
        intermediate_submaps = []
        intermediate_objmaps = []

        # For each layer m (except the last one)
        for m in range(self.num_layers- 1):
            # Get output from layer m
            hs_m = self.entity_outputs[m]
            hs_t = self.triplet_outputs[m]
            submap_m = self.submap_outputs[m]
            objmap_m = self.objmap_outputs[m]
            subject_m, object_m = torch.split(hs_t, self.reltr.hidden_dim, dim=-1)

            intermediate_entity.append(hs_m)
            intermediate_triplet.append(hs_t)
            intermediate_submaps.append(submap_m)
            intermediate_objmaps.append(objmap_m)


            intermediate_entity_early = intermediate_entity.copy()
            intermediate_triplet_early = intermediate_triplet.copy()
            intermediate_submaps_early = intermediate_submaps.copy()
            intermediate_objmaps_early = intermediate_objmaps.copy()

            # print("intermediate shape", hs_m.shape, hs_t.shape)

            # imit_hs_k = hs_m
            # imit_subject_k = subject_m
            # imit_object_k = object_m
            # imit_submap_k = submap_m
            # imit_objmap_k = objmap_m

            # print("stack shape test", torch.stack(intermediate_entity).shape)
            # print("stack shape test", torch.stack(intermediate_entity).transpose(1,2).shape)
            imit_hs_k = torch.mean(torch.stack(intermediate_entity), dim=0).transpose(0,1)
            temp = torch.mean(torch.stack(intermediate_triplet), dim=0).transpose(0,1)
            imit_subject_k,imit_object_k = torch.split(temp, self.reltr.hidden_dim, dim=-1)
            imit_submap_k = torch.mean(torch.stack(intermediate_submaps), dim=0)
            imit_objmap_k = torch.mean(torch.stack(intermediate_objmaps), dim=0)
            # print("transpose shapes -", imit_hs_k.shape, imit_subject_k.shape, imit_object_k.shape)

            layer_imitation_loss = 0.0

            # For each deeper layer k
            for k in range(m + 1, self.num_layers):
                # print("imit",m,k)
                # Get the ground truth deep representation
                entity_k = self.entity_outputs[k]
                hs_t = self.triplet_outputs[k]
                submap_k = self.submap_outputs[k]
                objmap_k = self.objmap_outputs[k]
                subject_k, object_k = torch.split(hs_t, self.reltr.hidden_dim, dim=-1)

                # _, _, hw = submap_k.shape
                # h = int(hw**0.5)  # Assume square
                # w = hw // h  # Ensure h*w = hw
                # imitated_entity_k = self.imitation_networks_enitity[k](hs_m)
                # imitated_subject_k = self.imitation_networks_subject[k](subject_m)
                # imitated_object_k = self.imitation_networks_object[k](object_m)
                # imitated_submap_k = self.imitation_networks_submap[k](submap_m, self.h,self.w)
                # imitated_objmap_k = self.imitation_networks_objmap[k](objmap_m, self.h,self.w)

                imit_hs_k = self.imitation_networks_enitity[k](imit_hs_k)
                imit_subject_k = self.imitation_networks_subject[k](imit_subject_k)
                imit_object_k = self.imitation_networks_object[k](imit_object_k)
                imit_submap_k = self.imitation_networks_submap[k](imit_submap_k, self.h,self.w)
                imit_objmap_k = self.imitation_networks_objmap[k](imit_objmap_k, self.h,self.w)
                # print("training shape -", imit_hs_k.shape, imit_subject_k.shape, imit_object_k.shape)


                # Compute cosine similarity loss
                entity_sim = 1 - F.cosine_similarity(imit_hs_k.transpose(0,1), entity_k.detach(), dim=-1).mean()
                subject_sim = 1 - F.cosine_similarity(imit_subject_k.transpose(0,1), subject_k.detach(), dim=-1).mean()
                object_sim = 1 - F.cosine_similarity(imit_object_k.transpose(0,1), object_k.detach(), dim=-1).mean()
                submap_sim = 1 - F.cosine_similarity(imit_submap_k, submap_k.detach(), dim=-1).mean()
                objmap_sim = 1 - F.cosine_similarity(imit_objmap_k, objmap_k.detach(), dim=-1).mean()

                # imitation_loss += subject_sim + object_sim + submap_sim + objmap_sim + entity_sim

                layer_imitation_loss += entity_sim + subject_sim + object_sim + submap_sim + objmap_sim


                intermediate_entity_early.append(imit_hs_k.transpose(0,1))
                intermediate_triplet_early.append(torch.cat((imit_subject_k.transpose(0,1), imit_object_k.transpose(0,1)), dim=-1))
                intermediate_submaps_early.append(imit_submap_k)
                intermediate_objmaps_early.append(imit_objmap_k)

            imitation_loss_total += layer_imitation_loss

            hs = torch.stack(intermediate_entity_early)
            hs_t = torch.stack(intermediate_triplet_early)
            sub_maps = torch.stack(intermediate_submaps_early)
            obj_maps = torch.stack(intermediate_objmaps_early)

            so_masks = torch.cat((sub_maps.reshape(sub_maps.shape[0], self.bs, sub_maps.shape[2], 1, self.h, self.w),
                      obj_maps.reshape(obj_maps.shape[0], self.bs, obj_maps.shape[2], 1, self.h, self.w)), dim=3) #decoder + imitation stack
            hs = hs.transpose(1, 2)
            hs_t = hs_t.transpose(1, 2)
            memory_permute = self.memory.permute(1, 2, 0).view(self.bs, self.c, self.h, self.w)
            so_masks = so_masks.detach()
            so_masks = self.reltr.so_mask_conv(so_masks.view(-1, 2, self.src_raw_shape[-2],self.src_raw_shape[-1])).view(hs_t.shape[0], hs_t.shape[1], hs_t.shape[2],-1)
            so_masks = self.reltr.so_mask_fc(so_masks)

            hs_sub, hs_obj = torch.split(hs_t, self.reltr.hidden_dim, dim=-1)

            outputs_class = self.reltr.entity_class_embed(hs)
            outputs_coord = self.reltr.entity_bbox_embed(hs).sigmoid()

            outputs_class_sub = self.reltr.sub_class_embed(hs_sub)
            outputs_coord_sub = self.reltr.sub_bbox_embed(hs_sub).sigmoid()
            # print("output class shape for sub", outputs_class_sub.shape)

            outputs_class_obj = self.reltr.obj_class_embed(hs_obj)
            outputs_coord_obj = self.reltr.obj_bbox_embed(hs_obj).sigmoid()
            # print("output class shape for obj", outputs_class_obj.shape)

            outputs_class_rel = self.reltr.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1))

            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
                    'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
                    'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
                    'rel_logits': outputs_class_rel[-1]}

            loss_dict = criterion(out, targets)
            # print("criterion", loss_dict)

            weight_dict = criterion.weight_dict
            layer_task_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            task_loss_total += layer_task_loss


        imitation_loss_avg = imitation_loss_total / (self.num_layers * (self.num_layers - 1) / 2)
        task_loss_avg = task_loss_total / (self.num_layers - 1)


        # return imitation_loss / (self.num_layers * (self.num_layers - 1)), losses
        return imitation_loss_avg, task_loss_avg